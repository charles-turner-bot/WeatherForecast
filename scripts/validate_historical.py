#!/usr/bin/env python
"""Run three fixed seasonal FCN/ERA5 cases and generate a reproducible scorecard."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx.config import ForecastConfig
from perthwx.forecast import run_forecast, open_forecast
from perthwx.validation import (
    fetch_truth, grid_scores, station_baseline_scores, station_pairs_with_persistence,
)
from perthwx.verification import Station, station_forecast, fetch_observations, score_pairs

CASES = ["2022-01-01", "2022-07-01", "2022-10-01"]


def verify_station(ds, directory, *, offline=False):
    station = Station()
    forecast = station_forecast(ds, station)
    path = directory / "observations.csv"
    if path.exists():
        obs = pd.read_csv(path, float_precision="round_trip")
    else:
        if offline:
            raise FileNotFoundError(f"Offline replay needs observations: {path}")
        obs = fetch_observations(forecast.valid_time, forecast.variable.unique(), station)
        obs["observation"] = pd.to_numeric(obs.observation, errors="coerce").astype("float64")
        obs.to_csv(path, index=False)
    pairs = station_pairs_with_persistence(forecast, obs)
    pairs.to_csv(directory / "station_pairs.csv", index=False)
    score_pairs(pairs).to_csv(directory / "station_all_pairs_scores.csv", index=False)
    return pairs, station_baseline_scores(pairs), station_baseline_scores(pairs, by_lead=True)


def case_figures(ds, truth, pairs, case, report):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, variable, label, offset in zip(axes, ["t2m", "ws10m"], ["Temperature (°C)", "Wind speed (m/s)"], [273.15, 0]):
        data = pairs.loc[pairs.variable == variable].sort_values("lead_hours")
        ax.plot(data.lead_hours, data.forecast - offset, label="FCN")
        ax.plot(data.lead_hours, data.observation - offset, ".-", label="Airport observations")
        ax.plot(data.lead_hours, data.persistence - offset, "--", label="Observation persistence")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Lead time (hours)")
    fig.suptitle(f"Perth Airport — FCN/ERA5 initialized {case} 00 UTC")
    fig.tight_layout()
    fig.savefig(report / f"{case}-station.png", dpi=110)
    plt.close(fig)

    # Shared color limits make the evolution and error scales comparable.
    fig, axes = plt.subplots(
        3, 3, figsize=(11, 9), constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for col, hours in enumerate([24, 120, 240]):
        lead = np.timedelta64(hours, "h")
        f = ds.t2m.isel(time=0).sel(lead_time=lead) - 273.15
        o = truth.t2m.sel(lead_time=lead) - 273.15
        for row, (field, title) in enumerate([(f, "FCN"), (o, "ERA5"), (f - o, "FCN minus ERA5")]):
            ax = axes[row, col]
            field.plot(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="RdBu_r" if row == 2 else "RdYlBu_r",
                vmin=-10 if row == 2 else 0, vmax=10 if row == 2 else 45,
                cbar_kwargs={"label": "°C"},
            )
            ax.coastlines(resolution="110m", linewidth=0.6)
            ax.plot(115.97610, -31.92751, "ko", markersize=3)
            ax.set_title(f"{title}, +{hours} h")
    fig.suptitle(f"Perth domain temperature — {case} 00 UTC; dot = airport")
    fig.savefig(report / f"{case}-maps.png", dpi=100)
    plt.close(fig)


def summary_figure(scores, report):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    labels = {"t2m": "t2m (K)", "t850": "t850 (K)", "z500": "z500 (m²/s²)", "msl": "MSLP (Pa)", "u10m": "u10m (m/s)", "v10m": "v10m (m/s)"}
    for ax, (variable, label) in zip(axes.flat, labels.items()):
        for model, style in [("FCN", "-"), ("persistence", "--")]:
            data = scores.loc[(scores.variable == variable) & (scores.model == model)]
            # Equal-case root-mean-square of per-case RMSE, not mean RMSE.
            pooled = data.assign(mse=data.rmse ** 2).groupby("lead_hours").mse.mean().pow(0.5)
            ax.plot(pooled.index / 24, pooled, style, label=model)
        ax.set_title(label)
        ax.set_xlabel("Lead (days)")
        ax.set_ylabel("Area-weighted RMSE")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Perth ERA5 verification — {scores.case.nunique()} equally weighted seasonal cases")
    fig.savefig(report / "era5_rmse.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=CASES, choices=CASES)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/historical"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase1"))
    parser.add_argument("--offline", action="store_true", help="Require saved forecasts/truth/observations; never fetch or run inference")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        parser.error("Case dates must be unique")
    if args.offline:
        # Cartopy otherwise silently downloads its coastline at render time.
        coastline = Path("shapefiles/natural_earth/physical/ne_110m_coastline.shp")
        roots = [cartopy.config["data_dir"], cartopy.config["pre_existing_data_dir"]]
        if not any((Path(root) / coastline).is_file() for root in roots):
            raise FileNotFoundError("Offline maps require cached Cartopy 110m coastlines; run validate online first")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    grid, station, station_lead, provenance = [], [], [], []
    for case in args.cases:
        directory = args.outdir / case.replace("-", "")
        path = directory / "forecast.zarr"
        cfg = ForecastConfig(init_time=case + "T00:00:00", out_path=str(path), lead_days=10, crop_output=True, device=args.device)
        marker = path / "perthwx-run.json"
        if path.exists():
            if not marker.exists():
                raise ValueError(f"Incomplete forecast: {path}. Move it aside before retrying.")
            saved = json.loads(marker.read_text())
            # Compare JSON forms because tuple bounds become lists in metadata.
            if saved["config"] != json.loads(json.dumps(asdict(cfg))):
                raise ValueError(f"Forecast configuration mismatch: {path}")
        else:
            if args.offline:
                raise FileNotFoundError(f"Offline replay needs forecast: {path}")
            run_forecast(cfg)
        print(f"Validating {case}", flush=True)
        with open_forecast(str(path)) as opened:
            ds = opened.load()
        if not all(bool(np.isfinite(ds[v]).all()) for v in ds):
            raise ValueError(f"Nonfinite forecast values in {case}")
        truth = fetch_truth(ds, directory / "truth", offline=args.offline)
        if not all(bool(np.isfinite(truth[v]).all()) for v in truth):
            raise ValueError(f"Nonfinite ERA5 truth in {case}")
        # Lead zero must reproduce directly supplied ERA5 variables, not forecast skill.
        initial_error = float(abs(ds.isel(time=0, lead_time=0).to_array() - truth.isel(lead_time=0).to_array()).max())
        if initial_error > 0.01:
            raise ValueError(f"Initial analysis differs from ERA5: {initial_error}")
        scored = grid_scores(ds, truth).assign(case=case)
        pairs, station_scores, lead_scores = verify_station(ds, directory, offline=args.offline)
        if not pairs.loc[pairs.lead_hours > 0, "matched"].any():
            raise ValueError(f"No positive-lead station observations for {case}")
        scored.to_csv(directory / "grid_scores.csv", index=False)
        station_scores.to_csv(directory / "station_scores.csv", index=False)
        grid.append(scored)
        station.append(station_scores.assign(case=case))
        station_lead.append(lead_scores.assign(case=case))
        case_figures(ds, truth, pairs, case, args.report_dir)
        provenance.append(dict(case=case, run=json.loads(marker.read_text()), initial_max_abs_error=initial_error,
            observations_sha256=hashlib.sha256((directory / "observations.csv").read_bytes()).hexdigest(),
            truth_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((directory / "truth").glob("*.nc"))}))
    grid = pd.concat(grid, ignore_index=True)
    grid.to_csv(args.report_dir / "grid_scores.csv", index=False)
    pd.concat(station, ignore_index=True).to_csv(args.report_dir / "station_scores.csv", index=False)
    pd.concat(station_lead, ignore_index=True).to_csv(args.report_dir / "station_scores_by_lead.csv", index=False)
    summary_figure(grid, args.report_dir)
    recipe_paths = [Path("pixi.lock"), *Path("src/perthwx").glob("*.py"), Path(__file__)]
    recipe_hashes = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in recipe_paths}
    # Earth2Studio pins the FCN package to an immutable Hugging Face revision.
    from earth2studio.models.px import FCN
    package = FCN.load_default_package()
    checkpoint_hashes = {}
    for name in ("fcn.mdlus", "global_means.npy", "global_stds.npy"):
        # During offline replay use only the existing local cache; do not resolve downloads.
        from earth2studio.models.auto import Package
        cached = Path(Package.default_cache("fcn")) / name
        if cached.exists():
            checkpoint_hashes[name] = hashlib.sha256(cached.read_bytes()).hexdigest()
    metadata = dict(recipe_sha256=recipe_hashes, checkpoint_sha256=checkpoint_hashes, checkpoint_root=package.root,
        created_utc=datetime.now(timezone.utc).isoformat(), cases=provenance,
        grid_scoring="cos(latitude) weighting; shared finite FCN/truth/initial-analysis mask; lead > 0",
        station_scoring="nearest cell and finite report +/-30 minutes; shared mask with latest observation at/before init persistence",
        case_selection="Fixed 2022 summer, winter, spring dates chosen before scores; not selected extreme events",
        limitations="Three cases, one station; no climatology/ACC; no GFS forecast baseline; derived RH clipped to 0-100 percent")
    (args.report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Validation complete: {args.report_dir}")


if __name__ == "__main__":
    main()
