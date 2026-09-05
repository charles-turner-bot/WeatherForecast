"""earth2studio forecast driver (T2).

Composes a data source, a prognostic model, and a Zarr IO backend into a
deterministic forecast over the Perth domain, using ``earth2studio.run``.

earth2studio imports are done lazily inside functions so that importing this
module (and ``perthwx.config``) stays cheap and side-effect-free.

Run it:
    pixi run forecast --model FCN3 --init 2022-01-01T00:00:00 --lead-days 10
    pixi run forecast --dry-run          # validate wiring without downloading weights
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr

from .config import PERTH_DOMAIN, ForecastConfig


def _model_registry() -> dict:
    """Map config model names to earth2studio prognostic classes (lazy import)."""
    from earth2studio.models.px import FCN3, SFNO, GraphCastOperational

    return {
        "FCN3": FCN3,
        "SFNO": SFNO,
        "GraphCastOperational": GraphCastOperational,
    }


def _data_source(source: str):
    """Construct an earth2studio data source (lazy import)."""
    from earth2studio.data import ARCO, GFS

    s = source.lower()
    if s in ("arco", "era5"):
        return ARCO()
    if s == "gfs":
        return GFS()
    raise ValueError(f"Unknown source {source!r}; use 'arco' or 'gfs'.")


def load_model(name: str, device: str):
    """Load a prognostic model onto ``device`` (downloads weights on first use)."""
    reg = _model_registry()
    if name not in reg:
        raise ValueError(f"Unknown model {name!r}; known: {list(reg)}")
    cls = reg[name]
    package = cls.load_default_package()
    model = cls.load_model(package)
    return model


def run_forecast(cfg: ForecastConfig, dry_run: bool = False) -> str | None:
    """Run one deterministic forecast and write it to ``cfg.out_path`` (Zarr).

    Args:
        cfg: The forecast configuration.
        dry_run: If True, construct/validate every component and resolve the model
            package handle, but do NOT download weights or run inference. Prints
            the plan and returns None.

    Returns:
        The output Zarr path, or None for a dry run.
    """
    import torch

    reg = _model_registry()
    if cfg.model not in reg:
        raise ValueError(f"Unknown model {cfg.model!r}; known: {list(reg)}")

    data = _data_source(cfg.source)
    out = Path(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    from earth2studio.io import ZarrBackend

    io = ZarrBackend(file_name=str(out), backend_kwargs={"overwrite": True})
    output_coords = OrderedDict(variable=np.array(cfg.variables))

    if dry_run:
        pkg = reg[cfg.model].load_default_package()
        print("DRY RUN — wiring OK, no weights downloaded, no inference.")
        print(f"  model        : {cfg.model}  (package: {pkg})")
        print(f"  data source  : {cfg.source}")
        print(f"  init_time    : {cfg.init_time}")
        print(f"  lead_days    : {cfg.lead_days}  -> nsteps={cfg.nsteps}")
        print(f"  variables    : {cfg.variables}")
        print(f"  device       : {cfg.device}")
        print(f"  out_path     : {out}")
        return None

    device = torch.device(cfg.device)
    model = load_model(cfg.model, cfg.device)

    from earth2studio import run

    run.deterministic(
        [cfg.init_time],
        cfg.nsteps,
        model,
        data,
        io,
        output_coords=output_coords,
        device=device,
    )
    return str(out)


def open_forecast(path: str) -> xr.Dataset:
    """Open a forecast Zarr store as an xarray Dataset."""
    return xr.open_zarr(path)


def crop_to_perth(
    ds: xr.Dataset | xr.DataArray,
    domain: dict[str, tuple[float, float]] = PERTH_DOMAIN,
) -> xr.Dataset | xr.DataArray:
    """Crop a global dataset/array to the Perth window (handles descending lat)."""
    lat0, lat1 = domain["lat"]
    lon0, lon1 = domain["lon"]
    ds = ds.sortby("lat")  # ensure ascending so slice(low, high) works
    return ds.sel(lat=slice(min(lat0, lat1), max(lat0, lat1)),
                  lon=slice(lon0, lon1))


def _parse_args(argv: list[str] | None = None) -> ForecastConfig:
    p = argparse.ArgumentParser(description="Run a Perth AI weather forecast.")
    p.add_argument("--model", default="FCN3")
    p.add_argument("--source", default="arco", help="arco (ERA5) | gfs")
    p.add_argument("--init", dest="init_time", default="2022-01-01T00:00:00")
    p.add_argument("--lead-days", type=float, default=10.0)
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--out", dest="out_path", default="outputs/forecast.zarr")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    kwargs = dict(
        model=a.model, source=a.source, init_time=a.init_time,
        lead_days=a.lead_days, out_path=a.out_path, device=a.device,
    )
    if a.variables:
        kwargs["variables"] = a.variables
    cfg = ForecastConfig(**kwargs)
    cfg._dry_run = a.dry_run  # type: ignore[attr-defined]
    return cfg


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    path = run_forecast(cfg, dry_run=getattr(cfg, "_dry_run", False))
    if path:
        print(f"Forecast written to {path}")


if __name__ == "__main__":
    main()
