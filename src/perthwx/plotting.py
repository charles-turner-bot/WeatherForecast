"""Plotting utilities for weather-forecast fields over Perth, Western Australia.

Thin wrappers over xarray's plotting API (which handles facets, per-panel titles
and shared colorbars out of the box), adding cartopy coastlines/features, the
Perth marker, and the Perth map extent. PlateCarree throughout.

`constrained_layout` is used so titles/colorbars are never clipped on save
(cartopy GeoAxes are incompatible with ``bbox_inches="tight"``).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PERTH_LON = 115.86
PERTH_LAT = -31.95
PERTH_EXTENT = [108.0, 125.0, -40.0, -20.0]  # [lon_min, lon_max, lat_min, lat_max]


def _new_geoaxes() -> "plt.Axes":
    # No constrained_layout: with cartopy's fixed aspect it doesn't reserve title
    # room, so titles clip. We reserve room explicitly via subplots_adjust.
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    return ax


def _decorate(ax: "plt.Axes", extent: list[float] = PERTH_EXTENT, mark: bool = True) -> "plt.Axes":
    """Add coastlines/features, gridlines, extent, and (optionally) the Perth marker."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([110, 115, 120, 125])
    gl.ylocator = mticker.FixedLocator([-40, -35, -30, -25, -20])
    if mark:
        ax.plot([PERTH_LON], [PERTH_LAT], marker="o", color="red", markersize=6,
                zorder=10, transform=ccrs.PlateCarree())
        ax.annotate("Perth", xy=(PERTH_LON, PERTH_LAT), xytext=(5, 5),
                    textcoords="offset points", fontsize=9, zorder=11)
    return ax


def _inpanel_label(ax: "plt.Axes", text: str, fontsize: int = 10) -> None:
    """Draw a label inside the top-left of a map panel on a small white box.

    cartopy GeoAxes fill their box under fixed aspect, so ``set_title`` clips;
    an in-panel anchored label always renders (a common weather-map convention).
    """
    ax.text(
        0.03, 0.96, text, transform=ax.transAxes, va="top", ha="left",
        fontsize=fontsize, zorder=20,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
    )


def _coord_label(value) -> str:
    """Human-readable label for a coordinate value (handles datetimes/timedeltas)."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.datetime_as_string(arr, unit="h").replace("T", " ") + "Z"
    if np.issubdtype(arr.dtype, np.timedelta64):
        return f"+{arr / np.timedelta64(1, 'h'):.0f} h"
    return str(value)


def plot_field(
    da: xr.DataArray,
    ax: "plt.Axes | None" = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    extent: list[float] = PERTH_EXTENT,
    mark: bool = True,
    add_colorbar: bool = True,
) -> "plt.Axes":
    """Plot a single 2-D field over Perth via ``xarray.DataArray.plot``.

    Args:
        da: A 2-D DataArray with dims "lat"/"lon".
        ax: Existing GeoAxes, or None to create one.
        cmap, vmin, vmax: Colour mapping.
        title: Axis title ("" clears xarray's auto-title).
        cbar_label: Colorbar label (used when add_colorbar is True).
        extent: Map extent. mark: whether to mark Perth.
        add_colorbar: Attach a colorbar to this axis.

    Returns:
        The GeoAxes.
    """
    if ax is None:
        ax = _new_geoaxes()
    mappable = da.plot(
        ax=ax, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False,
    )
    _decorate(ax, extent, mark)
    ax.set_title("")
    if title:
        _inpanel_label(ax, title)
    if add_colorbar:
        ax.get_figure().colorbar(mappable, ax=ax, label=cbar_label, shrink=0.8)
    return ax


def plot_field_grid(
    da: xr.DataArray,
    col: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ncols: int = 4,
    cbar_label: str | None = None,
    extent: list[float] = PERTH_EXTENT,
    title_coord: str | None = None,
) -> "mpl.figure.Figure":
    """Small-multiples over dimension ``col`` via xarray's faceted plotting.

    xarray handles the grid, shared colorbar and layout; we add cartopy features
    per panel and (optionally) title panels by ``title_coord`` (e.g. "valid_time"
    for dates instead of the raw lead-time offset).

    Returns:
        The Figure.
    """
    n = int(da.sizes[col])
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if vmin is None:
        vmin = float(da.min())
    if vmax is None:
        vmax = float(da.max())

    # Explicit grid layout (cartopy GeoAxes + xarray's auto-colorbar don't leave
    # room for titles). xarray draws each field; we place a shared colorbar in a
    # dedicated axis and reserve top/row space for the per-panel date titles.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.7 * ncols, 3.7 * nrows), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axf = axes.flatten()
    mappable = None
    label_coord = title_coord if title_coord is not None else col
    for i, ax in enumerate(axf):
        if i < n:
            mappable = da.isel({col: i}).plot(
                ax=ax, x="lon", y="lat", transform=ccrs.PlateCarree(),
                cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False,
            )
            _decorate(ax, extent, mark=False)
            _inpanel_label(ax, _coord_label(da[label_coord].values[i]))
        else:
            ax.set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.05,
                        hspace=0.35, wspace=0.12)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(mappable, cax=cax, label=cbar_label)
    return fig


def plot_wind(
    u: xr.DataArray,
    v: xr.DataArray,
    ax: "plt.Axes | None" = None,
    step: int = 3,
    title: str | None = None,
    extent: list[float] = PERTH_EXTENT,
    background_speed: bool = True,
) -> "plt.Axes":
    """Plot 10 m wind (quiver) with an optional xarray-drawn speed background."""
    if ax is None:
        ax = _new_geoaxes()
    if background_speed:
        speed = np.sqrt(u**2 + v**2)
        mappable = speed.plot(
            ax=ax, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="viridis",
            add_colorbar=False,
        )
        ax.get_figure().colorbar(mappable, ax=ax, label="wind speed (m/s)", shrink=0.8)
    u_sub = u.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    v_sub = v.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    ax.quiver(u_sub.lon, u_sub.lat, u_sub, v_sub,
              transform=ccrs.PlateCarree(), scale=100, width=0.002)
    _decorate(ax, extent, mark=True)
    ax.set_title("")
    if title:
        _inpanel_label(ax, title)
    return ax
