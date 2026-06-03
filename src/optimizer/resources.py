"""
Resource definitions for the CAISO portfolio optimizer.

Each class encodes both physical operating limits and the regulatory
constraints that become hard LP constraints in the dispatch model:

  - HydroResource  : FERC Part 12 license constraints (water release quotas,
                     environmental bypass flows, intraday ramp ceilings)
  - SolarResource  : zero-fuel dispatchable curtailment, irradiance-driven
                     availability profile
  - GasPeakerResource : heat-rate derived marginal cost, startup penalty,
                        NERC minimum up/down times
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Resource:
    """Base for every dispatchable resource in the portfolio."""
    name: str
    node: str                     # CAISO pricing node (e.g., SP15_GEN-APND)
    capacity_mw: float            # nameplate MW
    min_output_mw: float          # technical minimum when online
    marginal_cost_per_mwh: float  # $/MWh (fuel + VOM)
    ramp_rate_mw_min: float       # maximum ramp, MW/minute


@dataclass
class HydroResource(Resource):
    """
    Conventional hydro subject to FERC Part 12 license conditions.

    Regulatory mapping to LP constraints:
      daily_energy_budget_mwh  →  Σ_h dispatch[h] ≤ budget
          Proxy for total daily water release quota under the FERC license.
          Constrains the optimizer from running the unit flat at nameplate.

      min_flow_mw              →  dispatch[h] ≥ min_flow  ∀h
          Minimum environmental bypass flow mandated by the FERC license to
          protect downstream fishery habitat (FERC Order 2003, Part 12).

      max_ramp_pct_capacity    →  |dispatch[h] - dispatch[h-1]| ≤ ramp_limit
          Some FERC licenses cap the rate of reservoir drawdown. Modeled
          as a ramp-rate ceiling as a fraction of nameplate capacity.

    Reference: FERC Order 764 (intraday scheduling flexibility requires that
    any 15-minute schedule deviation be covered by ramp capability).
    """
    daily_energy_budget_mwh: float = 1_200.0  # MWh/day water release cap
    min_flow_mw: float = 20.0                # environmental flow floor, MW
    max_ramp_pct_capacity: float = 0.15      # ≤15% of nameplate per interval


@dataclass
class SolarResource(Resource):
    """
    Utility-scale solar PV. Zero fuel cost; output limited by irradiance.

    availability_profile is a 24-element list in [0, 1] representing the
    fraction of nameplate capacity achievable each hour. Set at runtime via
    set_availability() before the LP model is built.
    """
    availability_profile: list[float] = field(default_factory=list)

    def set_availability(self, profile: list[float]) -> None:
        if len(profile) != 24:
            raise ValueError("Availability profile must contain exactly 24 hourly values.")
        if any(v < 0.0 or v > 1.0 for v in profile):
            raise ValueError("All availability values must be in [0, 1].")
        self.availability_profile = list(profile)

    def effective_capacity_mw(self, hour: int) -> float:
        """MW available at this hour, accounting for irradiance curtailment."""
        if not self.availability_profile:
            return 0.0
        return self.capacity_mw * self.availability_profile[hour]


@dataclass
class GasPeakerResource(Resource):
    """
    Simple-cycle combustion turbine used for peak shaving.

    Marginal cost = heat_rate × fuel_cost + VOM.
    startup_cost_usd is applied as a flat penalty when the unit transitions
    from zero to any positive dispatch level (approximated in LP relaxation
    by including it only for hours where dispatch > threshold).

    NERC reliability standards impose minimum up/down times to prevent
    rapid cycling that stresses the transmission system (NERC BAL-001).
    """
    heat_rate_mmbtu_per_mwh: float = 10.5    # simple-cycle CT, typical range 9-11
    fuel_cost_per_mmbtu: float = 4.50         # $/MMBtu spot gas price
    vom_per_mwh: float = 5.0                  # variable O&M, $/MWh
    startup_cost_usd: float = 2_500.0         # per cold-start event
    min_up_hours: int = 2                     # NERC scheduling constraint
    min_down_hours: int = 1

    def computed_marginal_cost(self) -> float:
        """Recalculate marginal cost from current heat rate and fuel price."""
        return self.heat_rate_mmbtu_per_mwh * self.fuel_cost_per_mmbtu + self.vom_per_mwh


def build_portfolio() -> dict[str, Resource]:
    """
    Instantiate the 3-resource hypothetical CAISO portfolio.

    Nodes reference real CAISO pricing aggregation points:
      NP15_GEN-APND — Northern CA (PG&E territory)
      SP15_GEN-APND — Southern CA (SCE/SDG&E territory)
    """
    hydro = HydroResource(
        name="Hydro",
        node="NP15_GEN-APND",
        capacity_mw=150.0,
        min_output_mw=20.0,
        marginal_cost_per_mwh=15.0,   # $/MWh — low; water is sunk cost
        ramp_rate_mw_min=2.5,          # MW/min — fast for hydro
        daily_energy_budget_mwh=1_200.0,
        min_flow_mw=20.0,
    )

    solar = SolarResource(
        name="Solar PV",
        node="SP15_GEN-APND",
        capacity_mw=80.0,
        min_output_mw=0.0,             # can curtail to zero
        marginal_cost_per_mwh=0.0,     # zero fuel cost
        ramp_rate_mw_min=10.0,         # curtailment can happen instantly
    )

    gas = GasPeakerResource(
        name="Gas Peaker",
        node="SP15_GEN-APND",
        capacity_mw=200.0,
        min_output_mw=40.0,            # technical minimum when online
        marginal_cost_per_mwh=89.25,   # heat_rate(10.5) × fuel(4.50) + VOM(5)
        ramp_rate_mw_min=5.0,          # MW/min — typical CT ramp
        heat_rate_mmbtu_per_mwh=10.5,
        fuel_cost_per_mmbtu=4.50,
        vom_per_mwh=5.0,
        startup_cost_usd=2_500.0,
    )

    return {"hydro": hydro, "solar": solar, "gas": gas}
