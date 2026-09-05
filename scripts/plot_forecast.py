#!/usr/bin/env python
"""Render standard Perth product maps from a forecast Zarr store.

Usage:
    pixi run plot --zarr outputs/fcn_gfs_7day.zarr --outdir outputs/figures

Produces (headless): a daily 2 m-temperature small-multiples sequence, a 10 m
wind field, and an MSLP map, cropped to the Perth window.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx.forecast import open_forecast, crop_to_perth  # noqa: E402
from perthwx import plotting as P  # noqa: E402


def _hour_index(lead_time, hours: float) -> int:
    h = lead_time / np.timedelta64(1, "h")
    return int(np.argmin(np.abs(h - hours)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", default="outputs/fcn_gfs_7day.zarr")
    ap.add_argument("--outdir", default="outputs/figures")
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = crop_to_perth(open_forecast(a.zarr)).squeeze()
    lt = ds.lead_time.values
    init_dt = np.asarray(ds.time.values).ravel()[0]
    init = np.datetime_as_string(init_dt, unit="h").replace("T", " ") + "Z"
    # valid time per lead step (init + lead), as a coordinate for panel titles
    ds = ds.assign_coords(valid_time=("lead_time", init_dt + lt))

    def valid_str(i: int) -> str:
        return np.datetime_as_string(init_dt + lt[i], unit="h").replace("T", " ") + "Z"

    # 1. daily 2 m temperature sequence — panels titled by valid date
    t2m = ds["t2m"] - 273.15
    seq = t2m.isel(lead_time=slice(0, None, 4))
    fig = P.plot_field_grid(seq, col="lead_time", ncols=4, cmap="RdYlBu_r",
                            cbar_label="2 m temperature (°C)", title_coord="valid_time")
    fig.suptitle(f"FCN 2 m temperature over Perth — daily, init {init} (GFS)", y=0.97)
    fig.savefig(outdir / "perth_t2m_sequence.png", dpi=95, bbox_inches=None)

    # 2. 10 m wind at +72 h
    i = _hour_index(lt, 72)
    ax = P.plot_wind(ds["u10m"].isel(lead_time=i), ds["v10m"].isel(lead_time=i),
                     title=f"FCN 10 m wind — {valid_str(i)} (+72 h)")
    ax.get_figure().savefig(outdir / "perth_wind_72h.png", dpi=95)

    # 3. MSLP at +120 h
    i = _hour_index(lt, 120)
    ax = P.plot_field(ds["msl"].isel(lead_time=i) / 100.0, cmap="viridis",
                      title=f"FCN MSLP — {valid_str(i)} (+120 h)",
                      cbar_label="hPa")
    ax.get_figure().savefig(outdir / "perth_mslp_120h.png", dpi=95)

    print(f"wrote figures to {outdir}/")


if __name__ == "__main__":
    main()
