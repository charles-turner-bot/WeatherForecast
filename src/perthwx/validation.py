"""Historical validation against ERA5 and initial-analysis persistence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .metrics import latitude_weights, rmse, bias, mae
from .verification import score_pairs, match_observations


def fetch_truth(forecast: xr.Dataset, directory: Path, *, source=None, offline: bool = False) -> xr.Dataset:
    """Fetch one valid time at a time, saving small exact-grid NetCDF snapshots.

    Restarting reuses complete snapshots. ARCO's global compressed chunks are
    temporary: only the forecast domain is retained on disk, limiting storage.
    """
    if forecast.sizes["time"] != 1:
        raise ValueError("Historical case must contain one initialization")
    directory.mkdir(parents=True, exist_ok=True)
    init = forecast.time.values[0]
    frames = []
    variables = sorted(forecast.data_vars)
    for lead in forecast.lead_time.values:
        valid = init + lead
        stamp = np.datetime_as_string(valid, unit="h").replace("-", "").replace(":", "")
        path = directory / f"{stamp}.nc"
        if path.exists():
            with xr.open_dataset(path) as saved:
                frame = saved.load()
        else:
            if offline:
                raise FileNotFoundError(f"Offline replay needs truth snapshot: {path}")
            if source is None:
                from earth2studio.data import ARCO
                source = ARCO(cache=False, verbose=False)
            print(f"ERA5 truth {stamp}", flush=True)
            values = source([valid], variables).sel(lat=forecast.lat, lon=forecast.lon)
            frame = values.to_dataset(dim="variable").astype(np.float32)
            frame.attrs["source"] = "earth2studio.data.ARCO / ERA5"
            temporary = path.with_suffix(".tmp.nc")
            frame.to_netcdf(temporary)
            temporary.replace(path)
        if list(frame.time.values) != [valid] or set(frame.data_vars) != set(variables):
            raise ValueError(f"Truth snapshot mismatch: {path}")
        xr.align(frame.lat, forecast.lat, join="exact")
        xr.align(frame.lon, forecast.lon, join="exact")
        frames.append(frame)
    truth = xr.concat(frames, dim="time").rename(time="lead_time")
    return truth.assign_coords(lead_time=forecast.lead_time)


def grid_scores(forecast: xr.Dataset, truth: xr.Dataset) -> pd.DataFrame:
    """Cos(latitude)-weighted scores by lead, on the same finite mask for both models.

    Persistence holds each ERA5 field at initialization. No climatology-free
    substitute is labelled ACC. z500 is geopotential (m²/s²), not height.
    """
    if forecast.sizes.get("time") != 1:
        raise ValueError("Expected a single forecast initialization")
    fc = forecast.isel(time=0, drop=True)
    if set(fc.data_vars) != set(truth.data_vars):
        raise ValueError("Forecast and truth variables must match")
    fc, truth = xr.align(fc, truth, join="exact")
    if np.timedelta64(0, "h") not in fc.lead_time.values:
        raise ValueError("Lead zero is required for persistence")
    weights = latitude_weights(fc.lat)
    rows = []
    for variable in sorted(fc.data_vars):
        if set(fc[variable].dims) != {"lead_time", "lat", "lon"} or set(truth[variable].dims) != {"lead_time", "lat", "lon"}:
            raise ValueError("Scoring requires lead_time/lat/lon fields")
        initial = truth[variable].sel(lead_time=np.timedelta64(0, "h"), drop=True)
        for lead in fc.lead_time.values:
            hours = float(lead / np.timedelta64(1, "h"))
            if hours <= 0:
                continue
            f = fc[variable].sel(lead_time=lead, drop=True)
            o = truth[variable].sel(lead_time=lead, drop=True)
            mask = np.isfinite(f) & np.isfinite(o) & np.isfinite(initial)
            for name, prediction in (("FCN", f), ("persistence", initial)):
                row = dict(
                    variable=variable, lead_hours=hours, model=name,
                    n_expected=mask.size, n_matched=int(mask.sum()),
                )
                for metric in (rmse, bias, mae):
                    row[metric.__name__] = (
                        float(metric(prediction.where(mask), o.where(mask),
                                     dim=["lat", "lon"], lat_weights=weights))
                        if row["n_matched"] else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def station_baseline_scores(pairs: pd.DataFrame, by_lead: bool = False) -> pd.DataFrame:
    """Compare FCN and station-observation persistence on identical positive leads.

    Baseline is the latest finite observation at or before initialization (within
    the pairing tolerance), never a future report. It is matched separately by
    the caller, then saved as `persistence` alongside the observation pairs.
    """
    shared = pairs.copy()
    shared["matched"] = shared.matched & np.isfinite(shared.persistence)
    model = score_pairs(shared, by_lead=by_lead).assign(model="FCN")
    shared["forecast"] = shared.persistence
    baseline = score_pairs(shared, by_lead=by_lead).assign(model="persistence")
    return pd.concat([model, baseline], ignore_index=True)


def station_pairs_with_persistence(forecast: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Match one historical case, then attach an observation baseline without look-ahead."""
    if forecast.init_time.nunique() != 1:
        raise ValueError("Station persistence requires one initialization per case")
    pairs = match_observations(forecast, observations)
    observed_times = pd.to_datetime(observations.time, utc=True).dt.tz_localize(None)
    init = pd.Timestamp(forecast.init_time.iloc[0])
    eligible = observations.loc[observed_times <= init]
    initial = match_observations(forecast.loc[forecast.lead_hours == 0], eligible)
    initial = initial[["station", "variable", "observation", "observation_time"]].rename(
        columns={"observation": "persistence", "observation_time": "persistence_time"})
    return pairs.merge(initial, on=["station", "variable"], how="left", validate="many_to_one")
