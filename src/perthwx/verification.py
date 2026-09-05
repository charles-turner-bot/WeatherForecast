"""Match deterministic forecasts to IEM station observations in SI units.

Observation input follows earth2studio's long-form schema. Nearest finite
observation per variable within a bounded UTC window; ties choose the earlier
report. No temporal interpolation, filling, or pressure-altimeter substitution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from . import metrics

UNITS = {"t2m": "K", "u10m": "m s-1", "v10m": "m s-1", "ws10m": "m s-1", "msl": "Pa"}


@dataclass(frozen=True)
class Station:
    identifier: str = "YPPH"
    lat: float = -31.92751
    lon: float = 115.97610


def station_forecast(ds: xr.Dataset, station: Station) -> pd.DataFrame:
    """Extract the nearest grid cell; retain init, lead, and valid UTC times.

    Accepts earth2studio deterministic time/lead_time/lat/lon datasets. Rejects
    out-of-domain stations rather than silently selecting a distant boundary.
    Native earth2studio output is assumed to use SI units.
    """
    for coord in ("time", "lead_time", "lat", "lon"):
        if coord not in ds.coords or ds[coord].dims != (coord,) or not ds[coord].size:
            raise ValueError(f"Expected nonempty one-dimensional {coord} coordinate")
        if not ds.indexes[coord].is_unique:
            raise ValueError(f"Duplicate {coord} coordinates")
    if not np.issubdtype(ds.time.dtype, np.datetime64) or not np.issubdtype(ds.lead_time.dtype, np.timedelta64):
        raise ValueError("time must be datetime64 and lead_time must be timedelta64")
    if pd.isna(ds.time.values).any() or pd.isna(ds.lead_time.values).any() or (ds.lead_time.values < np.timedelta64(0, "h")).any():
        raise ValueError("Invalid time or negative lead_time")
    lon = station.lon % 360 if float(ds.lon.min()) >= 0 else (station.lon + 180) % 360 - 180
    if not (float(ds.lat.min()) <= station.lat <= float(ds.lat.max()) and float(ds.lon.min()) <= lon <= float(ds.lon.max())):
        raise ValueError("Station is outside forecast domain")
    names = [v for v in UNITS if v in ds]
    if not names:
        raise ValueError("Forecast has no supported surface variables")
    point = ds[names].sortby("lat").sortby("lon").sel(lat=station.lat, lon=lon, method="nearest")
    for name in names:
        if set(point[name].dims) != {"time", "lead_time"}:
            raise ValueError(f"{name}: expected deterministic time/lead_time dimensions")
    if "u10m" in point and "v10m" in point:
        point["ws10m"] = np.hypot(point.u10m, point.v10m)
        if "ws10m" not in names:
            names.append("ws10m")
    frame = point.to_dataframe().reset_index().rename(columns={"time": "init_time", "lat": "grid_lat", "lon": "grid_lon"})
    frame["valid_time"] = frame.init_time + frame.lead_time
    frame["lead_hours"] = frame.lead_time / pd.Timedelta(hours=1)
    frame = frame.melt(id_vars=["init_time", "valid_time", "lead_hours", "grid_lat", "grid_lon"], value_vars=names, var_name="variable", value_name="forecast")
    frame["station"] = station.identifier
    frame["station_lat"], frame["station_lon"] = station.lat, station.lon
    frame["units"] = frame.variable.map(UNITS)
    return frame


def fetch_observations(times, variables, station: Station, tolerance_minutes: float = 30) -> pd.DataFrame:
    """Fetch IEM reports in SI, skipping future times. CLI saves a replay snapshot.

    Disable upstream caching so a partial recent window is refreshed on rerun.
    """
    if not np.isfinite(tolerance_minutes) or tolerance_minutes < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    from earth2studio.data import IEM_ASOS

    requested = pd.DatetimeIndex(times).unique()
    requested = requested[requested <= pd.Timestamp.now(tz="UTC").tz_localize(None)]
    if not len(requested):
        return pd.DataFrame(columns=["station", "variable", "time", "observation"])
    source = IEM_ASOS(cache=False, stations=[station.identifier], time_tolerance=np.timedelta64(round(tolerance_minutes * 60), "s"), async_workers=1, async_timeout=120, retries=1)
    return source(requested.to_numpy(dtype="datetime64[ns]"), list(variables))


def match_observations(forecast: pd.DataFrame, observations: pd.DataFrame, tolerance_minutes: float = 30) -> pd.DataFrame:
    """Preserve every forecast row, including missing obs and nonfinite forecasts.

    Duplicate reports at the same station/time/variable are averaged. Matching
    is variable-specific; actual observation timestamps and offsets are saved.
    """
    if not np.isfinite(tolerance_minutes) or tolerance_minutes < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    required = {"station", "variable", "time", "observation"}
    if not required <= set(observations):
        raise ValueError(f"Observation columns missing: {sorted(required - set(observations))}")
    obs = observations.copy()
    obs["observation_time"] = pd.to_datetime(obs.time, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    obs["observation"] = pd.to_numeric(obs.observation, errors="coerce").astype("float64")
    obs = obs.loc[np.isfinite(obs.observation) & obs.observation_time.notna()]
    obs = obs.groupby(["station", "variable", "observation_time"], as_index=False).observation.mean()
    result = []
    for (station, variable), group in forecast.groupby(["station", "variable"], sort=True):
        right = obs.loc[(obs.station == station) & (obs.variable == variable), ["observation_time", "observation"]].sort_values("observation_time")
        left = group.copy()
        left["valid_time"] = pd.to_datetime(left.valid_time, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        result.append(pd.merge_asof(left.sort_values("valid_time"), right, left_on="valid_time", right_on="observation_time", direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes)))
    if not result:
        raise ValueError("No forecast rows to verify")
    pairs = pd.concat(result, ignore_index=True)
    pairs["offset_minutes"] = (pairs.observation_time - pairs.valid_time) / pd.Timedelta(minutes=1)
    pairs["matched"] = np.isfinite(pairs.forecast) & np.isfinite(pairs.observation)
    return pairs


def score_pairs(pairs: pd.DataFrame, by_lead: bool = False) -> pd.DataFrame:
    """RMSE, bias, MAE and pair counts; exclude lead zero from skill scores."""
    keys = ["station", "variable", "units"] + (["lead_hours"] if by_lead else [])
    rows = []
    for key, group in pairs.loc[pairs.lead_hours > 0].groupby(keys):
        valid = group.loc[group.matched]
        row = dict(zip(keys, key))
        row.update(n_expected=len(group), n_matched=len(valid), n_missing=len(group) - len(valid))
        f = xr.DataArray(valid.forecast.to_numpy(), dims="sample")
        o = xr.DataArray(valid.observation.to_numpy(), dims="sample")
        row.update({name: float(getattr(metrics, name)(f, o)) if len(valid) else np.nan for name in ("rmse", "bias", "mae")})
        rows.append(row)
    return pd.DataFrame(rows, columns=keys + ["n_expected", "n_matched", "n_missing", "rmse", "bias", "mae"])
