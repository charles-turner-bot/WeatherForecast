"""Smoke/structure tests for perthwx.plotting (headless).

Run: `pixi run test-plots`. Renders each function and checks the resulting
figure structure (map axis with a pcolormesh, colorbars) without writing files
(cartopy + bbox_inches='tight' clips GeoAxes, so we assert on objects instead).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx import plotting as P  # noqa: E402

_fails: list[str] = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        _fails.append(name)


def _field(seed: int) -> xr.DataArray:
    lat = np.arange(-40, -19.75, 0.25)
    lon = np.arange(108, 125.25, 0.25)
    rng = np.random.RandomState(seed)
    data = 20 + 5 * np.cos(np.deg2rad(lat))[:, None] + rng.randn(len(lat), len(lon))
    return xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


# single field: map axis + colorbar, mesh present, extent correct
ax = P.plot_field(_field(0), title="t2m", cbar_label="C")
check("plot_field has mesh", len(ax.collections) >= 1)
check("plot_field has colorbar axis", len(ax.get_figure().axes) == 2)
check("plot_field extent", ax.get_xlim() == (108.0, 125.0) and ax.get_ylim() == (-40.0, -20.0))
plt.close("all")

# grid over lead_time: n panels drawn, one shared colorbar, unused hidden
lt = np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]")
da = xr.concat([_field(i) for i in range(len(lt))], dim="lead_time").assign_coords(lead_time=lt)
fig = P.plot_field_grid(da, col="lead_time", ncols=3, cbar_label="C")
# count GeoAxes panels only (a shared colorbar Axes is also in fig.axes)
panels = [a for a in fig.axes if type(a).__name__ == "GeoAxes" and a.collections]
check("grid drew n panels", len(panels) == len(lt))
check("grid hid unused panel", len([a for a in fig.axes if not a.get_visible()]) == 1)  # 6 slots, 5 used
plt.close("all")

# wind: quiver + speed background
ax = P.plot_wind(_field(10) - 20, _field(11) - 20, title="wind")
check("wind has artists", len(ax.collections) >= 1)
plt.close("all")

print("\nRESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAILURES: {_fails}")
sys.exit(1 if _fails else 0)
