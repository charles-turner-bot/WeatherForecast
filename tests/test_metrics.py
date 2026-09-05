"""Known-answer tests for perthwx.metrics.

Standalone (no pytest needed): `pixi run test`. Exits non-zero on any failure.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx import metrics as m  # noqa: E402

_fails: list[str] = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        _fails.append(name)


def mk(vals, dims=("lat", "lon")) -> xr.DataArray:
    return xr.DataArray(np.asarray(vals, float), dims=dims)


truth = mk([[10, 12], [14, 16]])

# deterministic metrics ------------------------------------------------------
check("rmse perfect=0", float(m.rmse(truth, truth)) == 0)
check("mae perfect=0", float(m.mae(truth, truth)) == 0)
check("bias perfect=0", float(m.bias(truth, truth)) == 0)

fc = truth + 2
check("bias offset=2", abs(float(m.bias(fc, truth)) - 2) < 1e-9)
check("mae offset=2", abs(float(m.mae(fc, truth)) - 2) < 1e-9)
check("rmse offset=2", abs(float(m.rmse(fc, truth)) - 2) < 1e-9)

clim = mk([[11, 11], [15, 15]])
check("acc perfect=1", abs(float(m.acc(truth, truth, clim)) - 1) < 1e-9)

# latitude weights normalize to mean 1 --------------------------------------
lat = xr.DataArray([-30.0, 30.0], dims="lat")
check("latweights mean~1", abs(float(m.latitude_weights(lat).mean()) - 1) < 1e-9)

# ensemble metrics -----------------------------------------------------------
ens = xr.DataArray(np.full((3, 2, 2), 10.0), dims=("member", "lat", "lon"))
check("crps deterministic=|v-y|",
      abs(float(m.crps_ensemble(ens, truth)) - float(np.abs(10 - truth).mean())) < 1e-9)
check("crps perfect=0", abs(float(m.crps_ensemble(truth.expand_dims(member=3), truth))) < 1e-9)

ens_low = xr.DataArray(np.zeros((4, 2, 2)), dims=("member", "lat", "lon"))
h = m.rank_histogram(ens_low, truth)  # obs > all members -> last bin
check("rankhist shape=n+1", h.shape == (5,))
check("rankhist all last bin", h[-1] == 4 and h[:-1].sum() == 0)

ens_r = truth + xr.DataArray(np.random.RandomState(0).randn(5, 2, 2),
                             dims=("member", "lat", "lon"))
ss = float(m.spread_skill_ratio(ens_r, truth))
check("spreadskill finite>0", np.isfinite(ss) and ss > 0)

print("\nRESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAILURES: {_fails}")
sys.exit(1 if _fails else 0)
