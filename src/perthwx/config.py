"""Configuration for Perth forecasts (T1).

Small, explicit config object consumed by the forecast driver (``forecast.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

# Perth point of interest (degrees).
PERTH_LAT: float = -31.95
PERTH_LON: float = 115.86

# Crop window used on *read* (lat/lon inclusive ranges, degrees). Covers Perth,
# the SW WA corner, and the adjacent Indian Ocean where WA's weather comes from.
PERTH_DOMAIN: dict[str, tuple[float, float]] = {
    "lat": (-40.0, -20.0),
    "lon": (108.0, 125.0),
}

# Default v1 variable set (all present in FCN3/SFNO output).
DEFAULT_VARIABLES: list[str] = ["t2m", "u10m", "v10m", "msl", "z500", "t850"]

# Models known to the driver, with their native forecast step in hours. Used to
# convert a desired lead time in days into nsteps.
MODEL_STEP_HOURS: dict[str, int] = {
    "FCN": 6,          # classic FourCastNet (AFNO) — fits the P40 in FP32
    "Pangu24": 24,
    "Pangu6": 6,
    "Pangu3": 3,
    "FCN3": 6,
    "SFNO": 6,
    "GraphCastOperational": 6,
}


@dataclass
class ForecastConfig:
    """Everything needed to produce one deterministic forecast.

    Attributes:
        init_time: Initialisation time (ISO 8601). For ``source="arco"`` this is
            a historical ERA5 time (reproducible research); for ``source="gfs"``
            it must be a recent GFS cycle.
        model: One of ``MODEL_STEP_HOURS`` keys.
        source: ``"arco"`` (ERA5, research) or ``"gfs"`` (real-time).
        lead_days: Desired forecast length in days (converted to nsteps).
        variables: Variables to write (subset of the model's outputs).
        out_path: Zarr output path.
        device: Torch device string (e.g. ``"cuda:0"``). P40s run FP32.
        crop_output: Store only the domain while running global inference.
        domain: Crop window used on read or when crop_output is True.
    """

    init_time: str = "2022-01-01T00:00:00"
    model: str = "FCN"
    source: str = "arco"
    lead_days: float = 10.0
    variables: list[str] = field(default_factory=lambda: list(DEFAULT_VARIABLES))
    out_path: str = "outputs/forecast.zarr"
    device: str = "cuda:0"
    crop_output: bool = False
    domain: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(PERTH_DOMAIN)
    )

    @property
    def nsteps(self) -> int:
        """Number of model steps for ``lead_days`` at the model's native step."""
        step_h = MODEL_STEP_HOURS.get(self.model)
        if step_h is None:
            raise ValueError(
                f"Unknown model {self.model!r}; known: {list(MODEL_STEP_HOURS)}"
            )
        steps = self.lead_days * 24 / step_h
        if not math.isfinite(steps) or steps <= 0 or not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("lead_days must be positive and an exact multiple of the model step")
        return int(round(steps))
