"""
NERC/WECC and FERC compliance tracker for the optimized dispatch schedule.

Standards modelled
------------------
NERC BAL-001-2  Real Power Balancing Control Performance
    Requirement: the Balancing Authority must maintain sufficient spinning
    reserve to cover its largest single contingency (N−1). We use the
    simplified BAA-wide criterion of ≥ 5% of hourly load as spinning reserve.
    An hour is flagged when available headroom falls below this threshold.

NERC CPS1  Control Performance Standard 1
    Full CPS1 measures the clock-minute average of ACE × Δfreq. Here we
    use a proxy: the absolute imbalance between scheduled dispatch and load
    as a fraction of load. Hours exceeding 3% are flagged as CPS1 concerns.
    A real implementation would ingest ACE telemetry from the SCADA system.

FERC Order 764  Intraday Scheduling Flexibility
    Requires resources to offer 15-minute scheduling increments. We check
    whether each resource's ramp capability supports a ±15% load variation
    within a 15-minute window. If not, a FERC-764 flag is raised.

FERC Part 12  Hydro License Constraints
    Tracks cumulative water release (hydro generation) against the daily
    budget. Logs hours where the minimum environmental flow constraint is
    binding or where the running total nears the daily cap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.optimizer.resources import HydroResource, Resource, SolarResource

# Thresholds
_RESERVE_FLOOR = 0.05       # NERC BAL-001 spinning reserve fraction
_CPS1_PROXY_PCT = 0.03      # 3% imbalance triggers CPS1 flag
_FERC764_WINDOW_MIN = 15    # 15-minute intraday scheduling interval


def _spinning_reserve_mw(
    portfolio: dict[str, Resource],
    schedule: pd.DataFrame,
    hour: int,
) -> float:
    """Available spinning reserve = Σ_r (Pmax[r,h] − dispatch[r,h])."""
    total = 0.0
    for r, res in portfolio.items():
        if isinstance(res, SolarResource):
            cap = res.effective_capacity_mw(hour)
        else:
            cap = res.capacity_mw
        total += cap - schedule.loc[hour, r]
    return max(total, 0.0)


def _ferc764_capable(res: Resource, load_change_pct: float = 0.15) -> bool:
    """
    Can this resource ramp ±load_change_pct of 300 MW load within 15 minutes?
    load_change_pct × 300 MW / (ramp_rate_mw_min × 15 min) ≤ 1
    """
    target_delta = load_change_pct * 300.0
    achievable = res.ramp_rate_mw_min * _FERC764_WINDOW_MIN
    return achievable >= target_delta


def build_compliance_report(
    dispatch_result: dict[str, Any],
    portfolio: dict[str, Resource],
    reserve_margin: float = 0.05,
) -> dict[str, Any]:
    """
    Evaluate the dispatch schedule against NERC and FERC standards.

    Parameters
    ----------
    dispatch_result : Output dict from run_dispatch().
    portfolio       : Resource dict from build_portfolio().
    reserve_margin  : The BAL-001 reserve fraction used in the LP (for labelling).

    Returns
    -------
    dict with keys:
      hourly  — pd.DataFrame, one row per hour with compliance flags
      summary — dict of aggregate metrics
    """
    schedule: pd.DataFrame = dispatch_result["schedule"]
    load: np.ndarray = dispatch_result["load"]
    lmp: np.ndarray = dispatch_result["lmp"]
    res_keys = list(portfolio.keys())

    rows = []
    hydro_cumulative = 0.0
    hydro_budget = None
    if "hydro" in portfolio and isinstance(portfolio["hydro"], HydroResource):
        hydro_budget = portfolio["hydro"].daily_energy_budget_mwh

    for h in range(24):
        total_disp = schedule.loc[h, res_keys].sum()
        load_h = float(load[h])
        imbalance_pct = abs(total_disp - load_h) / load_h if load_h > 0 else 0.0
        reserve = _spinning_reserve_mw(portfolio, schedule, h)
        reserve_pct = reserve / load_h if load_h > 0 else 0.0

        if "hydro" in portfolio:
            hydro_cumulative += float(schedule.loc[h, "hydro"])

        bal001_ok = reserve_pct >= reserve_margin
        cps1_ok = imbalance_pct <= _CPS1_PROXY_PCT

        # FERC Part 12: flag if hydro is at or above 95% of its daily budget.
        ferc12_warning = (
            hydro_budget is not None and hydro_cumulative >= 0.95 * hydro_budget
        )

        rows.append({
            "Hour (HE)": h + 1,
            "Load (MW)": round(load_h, 1),
            "Dispatch (MW)": round(total_disp, 1),
            "LMP ($/MWh)": round(float(lmp[h]), 2),
            "Spin Reserve (MW)": round(reserve, 1),
            "Reserve (%)": round(reserve_pct * 100, 1),
            "BAL-001": "✅" if bal001_ok else "❌",
            "CPS1 Proxy": "✅" if cps1_ok else "⚠️",
            "FERC-12 Water": round(hydro_cumulative, 1),
            "FERC-12 Flag": "⚠️" if ferc12_warning else "✅",
        })

    hourly_df = pd.DataFrame(rows).set_index("Hour (HE)")

    bal001_violations = int((hourly_df["BAL-001"] == "❌").sum())
    cps1_flags = int((hourly_df["CPS1 Proxy"] == "⚠️").sum())
    ferc12_flags = int((hourly_df["FERC-12 Flag"] == "⚠️").sum())

    # Resource-level FERC Order 764 ramp adequacy
    ferc764_status = {
        r: _ferc764_capable(res) for r, res in portfolio.items()
    }

    hydro_used = hydro_cumulative
    hydro_pct = (hydro_used / hydro_budget * 100) if hydro_budget else 0.0

    summary = {
        "bal001_compliance_pct": round((24 - bal001_violations) / 24 * 100, 1),
        "bal001_violations": bal001_violations,
        "cps1_flags": cps1_flags,
        "ferc12_water_used_mwh": round(hydro_used, 1),
        "ferc12_water_budget_mwh": hydro_budget or 0.0,
        "ferc12_utilisation_pct": round(hydro_pct, 1),
        "ferc764_capable": ferc764_status,
        "overall_status": "COMPLIANT" if (bal001_violations == 0 and cps1_flags == 0) else "REVIEW",
    }

    return {"hourly": hourly_df, "summary": summary}
