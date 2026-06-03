"""
Price risk sensitivity analysis for the optimized dispatch schedule.

Approach
--------
After the LP is solved, the physical dispatch schedule is treated as fixed
(generation already committed to delivery). LMP prices are then perturbed
across N Monte Carlo scenarios to compute the distribution of portfolio
revenue. This mirrors how a generation portfolio manager stress-tests
a committed schedule against price uncertainty.

Key metrics
-----------
  Expected Revenue    : mean of the revenue distribution
  Revenue Std Dev     : standard deviation (1-sigma price risk)
  VaR-95             : 5th percentile of revenue (value at risk at 95% confidence)
  CVaR-95 (ES)       : conditional value at risk — mean of the worst 5% outcomes
                        (also called Expected Shortfall). CVaR is coherent and
                        convex; preferred over VaR for portfolio risk management.

Price scenario generation
-------------------------
Log-normal shocks: S_i = P_base × exp(σ × Z_i − σ²/2)  where Z_i ~ N(0,1)
The σ²/2 correction keeps the median scenario equal to the base forecast
(unbiased perturbation). Correlation across hours is intentionally ignored here;
a production system would use a covariance matrix estimated from historical CAISO
price data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_price_scenarios(
    base_lmp: np.ndarray,
    n_scenarios: int = 500,
    volatility: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a matrix of LMP price scenarios via log-normal perturbation.

    Parameters
    ----------
    base_lmp:    Shape-(24,) base price forecast ($/MWh).
    n_scenarios: Number of Monte Carlo draws.
    volatility:  Annual-equivalent price volatility as a fraction (e.g., 0.20 = 20%).
    seed:        RNG seed for reproducibility.

    Returns
    -------
    Shape-(n_scenarios, 24) array of scenario prices in $/MWh.
    All prices are floored at $1/MWh to prevent negative scenario prices.
    """
    rng = np.random.default_rng(seed)
    sigma = np.full(24, volatility)
    z = rng.standard_normal((n_scenarios, 24))
    # Log-normal: exp(σZ − σ²/2) has mean=1, preserving the base forecast in expectation.
    shocks = np.exp(sigma * z - 0.5 * sigma ** 2)
    return np.maximum(base_lmp[np.newaxis, :] * shocks, 1.0)


def sensitivity_analysis(
    schedule: pd.DataFrame,
    base_lmp: np.ndarray,
    n_scenarios: int = 500,
    volatility: float = 0.20,
    confidence_level: float = 0.95,
) -> dict:
    """
    Compute revenue risk metrics for the committed dispatch schedule.

    The schedule is held fixed; only prices vary across scenarios. Revenue
    is computed as total portfolio revenue (sum across all resources and hours).

    Returns
    -------
    dict with keys:
      expected_revenue   — mean daily revenue ($)
      revenue_std        — standard deviation ($)
      var_95             — 5th-percentile daily revenue; VaR at 95% confidence ($)
      cvar_95            — CVaR/Expected Shortfall at 95% confidence ($)
      revenue_p25/p75    — interquartile range ($)
      scenario_revenues  — shape-(n_scenarios,) array for histogram plotting
      price_scenarios    — shape-(n_scenarios, 24) for optional downstream use
      volatility_table   — DataFrame summarising metrics across 5 volatility levels
    """
    scenarios = generate_price_scenarios(base_lmp, n_scenarios, volatility)
    total_dispatch = schedule.sum(axis=1).to_numpy()  # shape (24,)

    # Vectorised revenue: (n_scenarios, 24) × (24,) → sum → (n_scenarios,)
    scenario_revenues = (scenarios * total_dispatch[np.newaxis, :]).sum(axis=1)

    tail_cut = int((1.0 - confidence_level) * n_scenarios)
    sorted_rev = np.sort(scenario_revenues)
    cvar = sorted_rev[:max(tail_cut, 1)].mean()

    metrics = {
        "expected_revenue": float(scenario_revenues.mean()),
        "revenue_std": float(scenario_revenues.std()),
        "var_95": float(np.percentile(scenario_revenues, 5)),
        "cvar_95": float(cvar),
        "revenue_p25": float(np.percentile(scenario_revenues, 25)),
        "revenue_p75": float(np.percentile(scenario_revenues, 75)),
        "scenario_revenues": scenario_revenues,
        "price_scenarios": scenarios,
    }

    # Build a sensitivity table across a fixed set of volatility levels.
    vol_levels = [0.05, 0.10, 0.20, 0.30, 0.40]
    rows = []
    for vol in vol_levels:
        sc = generate_price_scenarios(base_lmp, n_scenarios, vol, seed=42)
        rev = (sc * total_dispatch[np.newaxis, :]).sum(axis=1)
        tail = int((1 - confidence_level) * n_scenarios)
        rows.append({
            "Volatility": f"{int(vol * 100)}%",
            "Expected Revenue ($)": round(rev.mean()),
            "Std Dev ($)": round(rev.std()),
            "VaR-95 ($)": round(np.percentile(rev, 5)),
            "CVaR-95 ($)": round(np.sort(rev)[:max(tail, 1)].mean()),
        })

    metrics["volatility_table"] = pd.DataFrame(rows).set_index("Volatility")
    return metrics
