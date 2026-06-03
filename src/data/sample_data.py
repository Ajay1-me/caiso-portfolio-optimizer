"""
Synthetic CAISO-representative data for offline development and fallback.

LMP shape replicates the California "duck curve" — solar generation suppresses
midday prices while the rapid evening demand ramp drives 3–5× price spikes
between hours 17–20. This is the central scheduling challenge CAISO operators
face daily and the key driver for flexible resource dispatch.

Load profile is sized to match the 3-resource portfolio (430 MW nameplate):
  peak ~305 MW (evening), off-peak ~152 MW (overnight).
"""

from __future__ import annotations

import numpy as np
from datetime import date


# --- LMP $/MWh — 24-hour duck curve shape -----------------------------------
# Hour 0 = midnight, Hour 17-20 = evening peak, Hour 11-14 = solar belly
_DUCK_CURVE_BASE = np.array([
    28.0, 25.0, 23.0, 22.0, 24.0, 30.0,   # 00–05: overnight low
    48.0, 65.0, 72.0, 58.0, 42.0, 33.0,   # 06–11: morning ramp → solar onset
    28.0, 26.0, 25.0, 30.0, 58.0, 95.0,   # 12–17: solar belly → evening ramp
   130.0, 148.0, 140.0, 110.0, 75.0, 45.0, # 18–23: peak → decay
], dtype=float)

# --- MW load profile ---------------------------------------------------------
_LOAD_MW = np.array([
   165.0, 158.0, 155.0, 152.0, 155.0, 168.0,  # 00–05: overnight low
   195.0, 225.0, 248.0, 255.0, 252.0, 248.0,  # 06–11: morning ramp
   245.0, 240.0, 238.0, 242.0, 258.0, 280.0,  # 12–17: midday, shoulder
   295.0, 305.0, 300.0, 282.0, 255.0, 215.0,  # 18–23: evening peak, decay
], dtype=float)

# --- Solar availability profile (fraction of nameplate) ---------------------
# Represents a clear-day irradiance profile for Southern California.
_SOLAR_AVAIL = np.array([
   0.00, 0.00, 0.00, 0.00, 0.00, 0.02,   # 00–05: night / civil dawn
   0.10, 0.25, 0.45, 0.62, 0.75, 0.82,   # 06–11: morning ramp
   0.88, 0.90, 0.87, 0.80, 0.68, 0.50,   # 12–17: solar peak, afternoon decline
   0.30, 0.12, 0.02, 0.00, 0.00, 0.00,   # 18–23: sunset, night
], dtype=float)


def get_sample_lmp(op_date: str | date | None = None, noise_seed: int | None = None) -> np.ndarray:
    """
    Return a 24-element LMP array ($/MWh).

    Optionally adds date-seeded white noise (±8%) to produce unique daily
    shapes, which makes multi-day scenarios more realistic for risk analysis.
    """
    if op_date is not None:
        if isinstance(op_date, str):
            op_date = date.fromisoformat(op_date)
        seed = op_date.toordinal() if noise_seed is None else noise_seed
        rng = np.random.default_rng(seed)
        noise = 1.0 + rng.uniform(-0.08, 0.08, size=24)
        return np.maximum(_DUCK_CURVE_BASE * noise, 5.0)  # floor at $5/MWh

    return _DUCK_CURVE_BASE.copy()


def get_sample_load() -> np.ndarray:
    """Return the 24-element load profile (MW)."""
    return _LOAD_MW.copy()


def get_solar_availability() -> list[float]:
    """Return the 24-element solar capacity factor profile."""
    return _SOLAR_AVAIL.tolist()
