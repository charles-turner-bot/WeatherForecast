"""Weather-forecast verification metrics.

This module provides a collection of standard verification metrics for
weather forecasts, including bias, error statistics, anomaly correlation,
CRPS, rank histograms, and spread/skill ratios. All functions operate on
xarray.DataArray objects and support optional latitude weighting for
spatially reduced statistics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def latitude_weights(lat: xr.DataArray) -> xr.DataArray:
    """Compute normalized latitude weights for area-weighted averaging.

    The weights are proportional to the cosine of the latitude (in degrees),
    which approximates the relative area of latitude bands on a sphere. The
    resulting weights are normalized so that their mean is exactly 1.0.

    Args:
        lat: A DataArray containing latitude values in degrees.

    Returns:
        A DataArray named "lat_weights" with the same shape and coordinates
        as the input, containing normalized cosine-of-latitude weights.
    """
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    weights.name = "lat_weights"
    return weights


def bias(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    dim: str | list[str] | None = None,
    lat_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the mean error (bias) between forecast and truth.

    The bias is defined as the mean of (forecast - truth) over the specified
    dimensions. If latitude weights are provided, a weighted mean is used.

    Args:
        forecast: The forecast DataArray.
        truth: The observed/truth DataArray.
        dim: Dimension(s) over which to reduce. If None, reduce over all dimensions.
        lat_weights: Optional latitude weights for weighted averaging.

    Returns:
        A DataArray containing the mean error.
    """
    error = forecast - truth
    if lat_weights is not None:
        return error.weighted(lat_weights).mean(dim=dim)
    return error.mean(dim=dim)


def mae(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    dim: str | list[str] | None = None,
    lat_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the mean absolute error between forecast and truth.

    The MAE is defined as the mean of |forecast - truth| over the specified
    dimensions. If latitude weights are provided, a weighted mean is used.

    Args:
        forecast: The forecast DataArray.
        truth: The observed/truth DataArray.
        dim: Dimension(s) over which to reduce. If None, reduce over all dimensions.
        lat_weights: Optional latitude weights for weighted averaging.

    Returns:
        A DataArray containing the mean absolute error.
    """
    abs_error = np.abs(forecast - truth)
    if lat_weights is not None:
        return abs_error.weighted(lat_weights).mean(dim=dim)
    return abs_error.mean(dim=dim)


def rmse(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    dim: str | list[str] | None = None,
    lat_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the root mean squared error between forecast and truth.

    The RMSE is defined as sqrt(mean((forecast - truth)**2)) over the
    specified dimensions. If latitude weights are provided, the weighted mean
    of the squared error is computed before taking the square root.

    Args:
        forecast: The forecast DataArray.
        truth: The observed/truth DataArray.
        dim: Dimension(s) over which to reduce. If None, reduce over all dimensions.
        lat_weights: Optional latitude weights for weighted averaging.

    Returns:
        A DataArray containing the root mean squared error.
    """
    squared_error = (forecast - truth) ** 2
    if lat_weights is not None:
        mean_sq = squared_error.weighted(lat_weights).mean(dim=dim)
    else:
        mean_sq = squared_error.mean(dim=dim)
    return np.sqrt(mean_sq)


def acc(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    climatology: xr.DataArray,
    dim: str | list[str] | None = None,
    lat_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the Anomaly Correlation Coefficient (ACC).

    The ACC is computed as the correlation between forecast anomalies and
    truth anomalies, where anomalies are deviations from climatology.
    ACC = sum(fa * ta) / sqrt(sum(fa**2) * sum(ta**2)), where sums are
    computed over the specified dimensions using weighted means if weights
    are provided.

    Args:
        forecast: The forecast DataArray.
        truth: The observed/truth DataArray.
        climatology: The climatology DataArray (same shape as forecast/truth).
        dim: Dimension(s) over which to reduce. If None, reduce over all dimensions.
        lat_weights: Optional latitude weights for weighted averaging.

    Returns:
        A DataArray containing the anomaly correlation coefficient.
    """
    fa = forecast - climatology
    ta = truth - climatology

    def _weighted_mean(da: xr.DataArray) -> xr.DataArray:
        if lat_weights is not None:
            return da.weighted(lat_weights).mean(dim=dim)
        return da.mean(dim=dim)

    numerator = _weighted_mean(fa * ta)
    denom_fa = _weighted_mean(fa ** 2)
    denom_ta = _weighted_mean(ta ** 2)
    denominator = np.sqrt(denom_fa * denom_ta)
    return numerator / denominator


def crps_ensemble(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    member_dim: str = "member",
    dim: str | list[str] | None = None,
    lat_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the Continuous Ranked Probability Score (CRPS) for an ensemble.

    Uses the energy form: CRPS = mean_i |X_i - y| - 0.5 * mean_{i,j} |X_i - X_j|.
    The per-point CRPS is first computed by reducing over the member dimension
    and member pairs, then averaged over the specified dimensions (with
    optional latitude weighting).

    Args:
        forecast: The ensemble forecast DataArray with a member dimension.
        truth: The observed/truth DataArray (no member dimension).
        member_dim: The name of the ensemble member dimension.
        dim: Dimension(s) over which to average the per-point CRPS. If None, reduce over all dimensions.
        lat_weights: Optional latitude weights for weighted averaging.

    Returns:
        A DataArray containing the CRPS.
    """
    # First term: mean over members of |X_i - y|
    abs_diff_truth = np.abs(forecast - truth)
    term1 = abs_diff_truth.mean(dim=member_dim)

    # Second term: 0.5 * mean over member pairs of |X_i - X_j|
    # Broadcast member dimension against a renamed copy
    forecast_j = forecast.rename({member_dim: f"{member_dim}_j"})
    abs_diff_pair = np.abs(forecast - forecast_j)
    # Mean over both member dimensions
    term2 = 0.5 * abs_diff_pair.mean(dim=[member_dim, f"{member_dim}_j"])

    crps_pointwise = term1 - term2

    if lat_weights is not None:
        return crps_pointwise.weighted(lat_weights).mean(dim=dim)
    return crps_pointwise.mean(dim=dim)


def rank_histogram(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    member_dim: str = "member",
) -> np.ndarray:
    """Compute the Talagrand rank histogram for an ensemble forecast.

    The rank of the observation is the number of ensemble members less than
    the observation. The histogram counts how many observations fall into
    each rank bin (0 to n_members inclusive).

    Args:
        forecast: The ensemble forecast DataArray with a member dimension.
        truth: The observed/truth DataArray (no member dimension).
        member_dim: The name of the ensemble member dimension.

    Returns:
        A 1-D numpy array of length (n_members + 1) containing the counts
        in each rank bin.
    """
    # Rank of the observation = number of ensemble members strictly less than it.
    # xarray broadcasts `truth` (no member dim) across the member dim automatically,
    # so no manual expand_dims is needed.
    ranks = (forecast < truth).sum(dim=member_dim)

    # Flatten over all non-member dimensions
    ranks_flat = ranks.values.flatten()

    n_members = forecast.sizes[member_dim]
    # Rank can be 0 to n_members
    # Use bincount with minlength = n_members + 1
    histogram = np.bincount(ranks_flat.astype(int), minlength=n_members + 1)
    return histogram


def spread_skill_ratio(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    member_dim: str = "member",
    dim: str | list[str] | None = None,
    lat_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the ensemble spread/skill ratio.

    Spread is the square root of the ensemble variance (variance over the
    member dimension), averaged over the specified dimensions. Skill is the
    RMSE of the ensemble mean versus truth, also averaged over the specified
    dimensions. The ratio is spread / skill.

    Args:
        forecast: The ensemble forecast DataArray with a member dimension.
        truth: The observed/truth DataArray (no member dimension).
        member_dim: The name of the ensemble member dimension.
        dim: Dimension(s) over which to reduce. If None, reduce over all dimensions.
        lat_weights: Optional latitude weights for weighted averaging.

    Returns:
        A DataArray containing the spread/skill ratio.
    """
    # Ensemble mean
    ensemble_mean = forecast.mean(dim=member_dim)

    # Spread: sqrt(variance over member_dim), then average over dim
    ensemble_var = forecast.var(dim=member_dim)
    if lat_weights is not None:
        spread = np.sqrt(ensemble_var.weighted(lat_weights).mean(dim=dim))
    else:
        spread = np.sqrt(ensemble_var.mean(dim=dim))

    # Skill: RMSE of ensemble mean vs truth
    skill = rmse(ensemble_mean, truth, dim=dim, lat_weights=lat_weights)

    return spread / skill