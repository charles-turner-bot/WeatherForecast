#!/usr/bin/env python
"""Verify a forecast against Perth Airport IEM observations, or replay a saved CSV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx.forecast import open_forecast
from perthwx.verification import Station, station_forecast, fetch_observations, match_observations, score_pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", default="outputs/fcn_gfs_7day.zarr")
    p.add_argument("--outdir", default="outputs/verification")
    p.add_argument("--observations", help="Replay earth2studio long-form SI CSV; no network access")
    p.add_argument("--tolerance-minutes", type=float, default=30)
    a = p.parse_args(argv)
    station = Station()
    with open_forecast(a.zarr) as ds:
        forecast = station_forecast(ds, station)
    observations = (pd.read_csv(a.observations, float_precision="round_trip") if a.observations else fetch_observations(forecast.valid_time, forecast.variable.unique(), station, a.tolerance_minutes))
    # Preserve the exact float32 source values when serializing the CSV snapshot.
    observations["observation"] = pd.to_numeric(observations.observation, errors="coerce").astype("float64")
    pairs = match_observations(forecast, observations, a.tolerance_minutes)
    scores = score_pairs(pairs)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    observations.to_csv(out / "observations.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    score_pairs(pairs, by_lead=True).to_csv(out / "scores_by_lead.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "forecast": str(Path(a.zarr).resolve()), "station": station.__dict__,
        "observation_source": a.observations or "earth2studio.data.IEM_ASOS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance_minutes": a.tolerance_minutes,
        "spatial_matching": "nearest grid cell; no elevation correction",
        "temporal_matching": "nearest finite per variable; earlier on ties; duplicate timestamp mean",
        "scores": "lead > 0 only; SI units; finite pairs only",
    }, indent=2) + "\n")
    print(scores.to_string(index=False))
    print(f"Wrote verification to {out}")
    if not pairs.loc[pairs.lead_hours > 0, "matched"].any():
        raise SystemExit("No usable forecast/observation pairs beyond lead zero; see saved coverage counts.")


if __name__ == "__main__":
    main()
