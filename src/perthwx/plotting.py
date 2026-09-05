"""Plotting utilities for weather-forecast fields over Perth, Western Australia.

Visualise 2-D meteorological fields (temperature, pressure, wind) using Cartopy
and Matplotlib on the PlateCarree projection, centred on the Perth region.

Drafted with the local qwen model, then reviewed/fixed on integration:
- `plot_field` now resolves the figure when an existing axis is passed (qwen's
  draft raised NameError on `fig`);
- added an `add_colorbar` flag so grid panels share one colorbar;
- `plot_field_grid` was rewritten cleanly (the draft left reasoning comments and
  a broken colorbar-removal hack).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PERTH_LON = 115.86
PERTH_LAT = -31.95
PERTH_EXTENT = [108.0, 125.0, -40.0, -20.0]  # [lon_min, lon_max, lat_min, lat_max]


def _basemap(ax: "plt.Axes", extent: list[float] = PERTH_EXTENT) -> "plt.Axes":
    """Add coastlines, borders, states, and gridlines to a GeoAxes.

    Args:
        ax: The Cartopy GeoAxes to configure.
        extent: Map extent [lon_min, lon_max, lat_min, lat_max].

    Returns:
        The configured GeoAxes.
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    return ax


def mark_perth(ax: "plt.Axes", label: bool = True) -> "plt.Axes":
    """Mark the location of Perth on a GeoAxes.

    Args:
        ax: The Cartopy GeoAxes to plot on.
        label: Whether to annotate the marker with the text "Perth".

    Returns:
        The GeoAxes with the marker added.
    """
    ax.plot(
        [PERTH_LON], [PERTH_LAT], marker="o", color="red", markersize=6,
        zorder=10, transform=ccrs.PlateCarree(),
    )
    if label:
        ax.annotate(
            "Perth", xy=(PERTH_LON, PERTH_LAT), xytext=(5, 5),
            textcoords="offset points", fontsize=9, zorder=11,
        )
    return ax


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
    """Plot a single 2-D field over the Perth region.

    Args:
        da: A 2-D DataArray with dimensions "lat" and "lon".
        ax: Existing GeoAxes, or None to create a new figure/axes.
        cmap: Colormap name.
        vmin, vmax: Colour limits.
        title: Optional axis title.
        cbar_label: Colorbar label (used when add_colorbar is True).
        extent: Map extent.
        mark: Whether to mark Perth.
        add_colorbar: Whether to attach a colorbar to this axis.

    Returns:
        The GeoAxes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig = ax.get_figure()

    mesh = ax.pcolormesh(
        da.lon, da.lat, da, transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    _basemap(ax, extent=extent)
    if mark:
        mark_perth(ax)
    if add_colorbar:
        fig.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.8)
    if title:
        ax.set_title(title)
    return ax


def _coord_label(value) -> str:
    """Human-readable panel title for a coordinate value (handles datetimes)."""
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.datetime_as_string(arr, unit="h")
    if np.issubdtype(arr.dtype, np.timedelta64):
        return f"+{arr / np.timedelta64(1, 'h'):.0f} h"
    return str(value)


def plot_field_grid(
    da: xr.DataArray,
    col: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    ncols: int = 4,
    cbar_label: str | None = None,
    extent: list[float] = PERTH_EXTENT,
) -> "mpl.figure.Figure":
    """Small-multiples: one Perth panel per value along dimension ``col``.

    All panels share a colour scale and a single figure-level colorbar.

    Args:
        da: DataArray with a dimension ``col`` (e.g. "lead_time" or "time") plus
            "lat"/"lon".
        col: Dimension to iterate over.
        cmap: Colormap name.
        vmin, vmax: Shared colour limits; derived from ``da`` if None.
        ncols: Number of columns.
        cbar_label: Shared colorbar label.
        extent: Map extent.

    Returns:
        The Figure containing the grid.
    """
    n = int(da.sizes[col])
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    if vmin is None:
        vmin = float(da.min())
    if vmax is None:
        vmax = float(da.max())

    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(4.0 * ncols, 3.2 * nrows),
    )
    axes_flat = axes.flatten()

    last_ax = None
    for i in range(n):
        plot_field(
            da.isel({col: i}), ax=axes_flat[i], cmap=cmap, vmin=vmin, vmax=vmax,
            title=_coord_label(da[col].values[i]), extent=extent,
            mark=False, add_colorbar=False,
        )
        last_ax = axes_flat[i]
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if last_ax is not None and last_ax.collections:
        fig.colorbar(last_ax.collections[0], ax=axes, label=cbar_label, shrink=0.8)
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
    """Plot 10 m wind (quiver) with an optional wind-speed background.

    Args:
        u: Eastward wind component (m/s), dims "lat"/"lon".
        v: Northward wind component (m/s), dims "lat"/"lon".
        ax: Existing GeoAxes, or None to create a new figure/axes.
        step: Subsampling stride for the quiver arrows.
        title: Optional axis title.
        extent: Map extent.
        background_speed: Whether to shade wind speed behind the arrows.

    Returns:
        The GeoAxes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig = ax.get_figure()

    if background_speed:
        speed = np.sqrt(u**2 + v**2)
        mesh = ax.pcolormesh(
            u.lon, u.lat, speed, transform=ccrs.PlateCarree(),
            cmap="viridis", shading="auto",
        )
        fig.colorbar(mesh, ax=ax, label="wind speed (m/s)", shrink=0.8)

    u_sub = u.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    v_sub = v.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    ax.quiver(
        u_sub.lon, u_sub.lat, u_sub, v_sub,
        transform=ccrs.PlateCarree(), scale=100, width=0.002,
    )

    _basemap(ax, extent=extent)
    mark_perth(ax)
    if title:
        ax.set_title(title)
    return ax
