"""
24-hour Day-Ahead economic dispatch via linear programming (PuLP/CBC).

Model framing
-------------
This is a pure Economic Dispatch (ED) formulation — unit commitment (on/off
binary decisions) is treated as pre-determined. The LP decides how much power
each already-committed resource produces each hour at minimum cost.

For the gas peaker this means the LP can freely set dispatch to zero (treating
it as "not online"), which is the standard ED relaxation used in production cost
models before a full MILP UC solve.

Objective
---------
Minimize total variable production cost:
  min  Σ_h Σ_r  MC[r] × p[r,h]

Subject to:
  1. Load balance            Σ_r p[r,h] ≥ load[h]                ∀h
  2. Capacity ceiling        p[r,h] ≤ Pmax[r,h]                  ∀r,h
  3. Ramp limits             |p[r,h] − p[r,h−1]| ≤ ramp_hr[r]   ∀r, h>0
  4. FERC hydro budget       Σ_h p[hydro,h] ≤ daily_budget        (FERC Part 12)
  5. FERC min env. flow      p[hydro,h] ≥ min_flow                ∀h
  6. NERC BAL-001 reserve    Σ_r (Pmax[r,h] − p[r,h]) ≥ α×load[h] ∀h

Constraint (6) is the simplified spinning-reserve requirement from NERC
Reliability Standard BAL-001-2. The default α=5% is a common BAA (Balancing
Authority Area) planning criterion; real CAISO uses dynamic reserve targets
based on the largest single contingency (N−1).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

from src.optimizer.resources import GasPeakerResource, HydroResource, Resource, SolarResource

log = logging.getLogger(__name__)

HOURS = list(range(24))


def _capacity_at_hour(res: Resource, hour: int) -> float:
    """Effective nameplate capacity for resource *res* at *hour*."""
    if isinstance(res, SolarResource):
        return res.effective_capacity_mw(hour)
    return res.capacity_mw


def _lower_bound(res: Resource) -> float:
    """Minimum dispatch lower bound (applied uniformly across all hours)."""
    if isinstance(res, HydroResource):
        return res.min_flow_mw   # FERC environmental flow requirement
    return 0.0                   # all others can be at zero (ED relaxation)


def run_dispatch(
    portfolio: dict[str, Resource],
    lmp_prices: np.ndarray,
    load_profile: np.ndarray,
    reserve_margin: float = 0.05,
) -> dict[str, Any]:
    """
    Solve the 24-hour economic dispatch LP.

    Parameters
    ----------
    portfolio:
        Dict mapping resource keys to Resource instances (from build_portfolio).
    lmp_prices:
        Shape-(24,) array of Day-Ahead LMP prices in $/MWh.
    load_profile:
        Shape-(24,) array of hourly load obligations in MW.
    reserve_margin:
        Fraction of hourly load that must be held as spinning reserve (NERC BAL-001).
        Default 5% — adjust via the Streamlit sidebar.

    Returns
    -------
    dict with keys:
      schedule   — pd.DataFrame (24 × n_resources) of optimized dispatch in MW
      cost_usd   — total production cost ($)
      revenue_usd — total revenue at LMP (dispatch × LMP, $)
      status     — solver status string ('Optimal', 'Infeasible', …)
      lmp        — input LMP array (passed through for downstream callers)
      load       — input load array (passed through)
      reserve_margin — passed through for compliance tracker
    """
    res_keys = list(portfolio.keys())
    model = LpProblem("CAISO_DayAhead_ED", LpMinimize)

    # ── Decision variables ────────────────────────────────────────────────────
    # p[r, h] ∈ [lb, ub] where lb/ub encode physical and regulatory limits.
    p: dict[tuple[str, int], LpVariable] = {}
    for r in res_keys:
        res = portfolio[r]
        lb = _lower_bound(res)
        for h in HOURS:
            ub = _capacity_at_hour(res, h)
            # If lb > ub (e.g., solar at night: ub=0 but lb=0), clamp lb.
            effective_lb = min(lb, ub)
            p[(r, h)] = LpVariable(f"p_{r}_{h}", lowBound=effective_lb, upBound=ub)

    # ── Objective: minimize total production cost ─────────────────────────────
    model += (
        lpSum(
            portfolio[r].marginal_cost_per_mwh * p[(r, h)]
            for r in res_keys
            for h in HOURS
        ),
        "TotalProductionCost",
    )

    # ── Per-hour constraints ──────────────────────────────────────────────────
    for h in HOURS:
        load_h = float(load_profile[h])

        # (1) Load balance — must serve the full load each hour.
        model += (
            lpSum(p[(r, h)] for r in res_keys) >= load_h,
            f"LoadBalance_h{h}",
        )

        # (6) NERC BAL-001: spinning reserve = headroom above current dispatch.
        # Σ_r (Pmax[r,h] − p[r,h]) ≥ reserve_margin × load[h]
        model += (
            lpSum(
                _capacity_at_hour(portfolio[r], h) - p[(r, h)]
                for r in res_keys
            ) >= reserve_margin * load_h,
            f"SpinReserve_h{h}",
        )

    # ── FERC Part 12 hydro constraints ───────────────────────────────────────
    if "hydro" in portfolio:
        hydro: HydroResource = portfolio["hydro"]  # type: ignore[assignment]

        # (4) Daily energy budget — total generation ≤ water release quota.
        model += (
            lpSum(p[("hydro", h)] for h in HOURS) <= hydro.daily_energy_budget_mwh,
            "FERC_DailyEnergyBudget",
        )
        # Lower bound already set per-variable above; constraint (5) is implicit.

    # ── Ramp rate constraints ─────────────────────────────────────────────────
    # Convert MW/minute → MW/hour to match hourly dispatch intervals.
    for r in res_keys:
        ramp_hr = portfolio[r].ramp_rate_mw_min * 60.0  # MW/hour
        for h in range(1, 24):
            model += (p[(r, h)] - p[(r, h - 1)] <= ramp_hr, f"RampUp_{r}_h{h}")
            model += (p[(r, h - 1)] - p[(r, h)] <= ramp_hr, f"RampDn_{r}_h{h}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = PULP_CBC_CMD(msg=0)  # suppress CBC stdout
    status_code = model.solve(solver)
    status_str = LpStatus[status_code]
    log.info("Dispatch solver status: %s", status_str)

    if status_str != "Optimal":
        log.warning("Non-optimal solution: %s — returning zeros.", status_str)
        schedule = pd.DataFrame(
            np.zeros((24, len(res_keys))),
            columns=res_keys,
            index=pd.RangeIndex(24, name="hour"),
        )
        return {
            "schedule": schedule,
            "cost_usd": 0.0,
            "revenue_usd": 0.0,
            "margin_usd": 0.0,
            "status": status_str,
            "lmp": lmp_prices,
            "load": load_profile,
            "reserve_margin": reserve_margin,
        }

    # ── Extract results ───────────────────────────────────────────────────────
    schedule = pd.DataFrame(
        {r: [max(value(p[(r, h)]) or 0.0, 0.0) for h in HOURS] for r in res_keys},
        index=pd.RangeIndex(24, name="hour"),
    )

    total_dispatch = schedule.sum(axis=1).to_numpy()
    cost_usd = float(value(model.objective))
    revenue_usd = float((total_dispatch * lmp_prices).sum())

    return {
        "schedule": schedule,
        "cost_usd": cost_usd,
        "revenue_usd": revenue_usd,
        "margin_usd": revenue_usd - cost_usd,
        "status": status_str,
        "lmp": lmp_prices,
        "load": load_profile,
        "reserve_margin": reserve_margin,
    }
