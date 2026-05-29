"""
processing/speed.py — Pure-numpy utilities for speed data aggregation and NaN handling.

These functions are free of I/O and plotting side-effects; they only transform arrays.
"""

import numpy as np


def mean_over_sessions(values, min=0, max=None):
    """
    Session-averaged mean of `values` (axis 0) over the subset [min, max+1).

    NaN-aware: cells with no valid observations return NaN rather than 0,
    so downstream callers never confuse "no data" with "zero speed".
    """
    if max is None:
        max = values.shape[0]

    subset = values[min:max+1]
    valid_counts = np.sum(~np.isnan(subset), axis=0)
    summed = np.nansum(subset, axis=0)

    return np.divide(
        summed,
        valid_counts,
        out=np.full(summed.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )


def fill_trailing_nans_as_freeflow(profile, freeflow_value=1.0):
    """
    For each segment, fill NaNs that appear *after the last valid observation*
    with `freeflow_value`, treating end-of-simulation gaps as uncongested.

    Only trailing NaNs are touched; interior NaNs are left for fill_speed_nans.

    Parameters
    ----------
    profile        : ndarray (T, N)
    freeflow_value : float or ndarray (N,) — value to fill trailing NaNs with.
                     Use 1.0 for normalised r; use a per-segment speed (m/s) for raw.

    Returns
    -------
    ndarray (T, N), same dtype as input
    """
    filled = profile.copy()
    fv = (np.full(filled.shape[1], freeflow_value)
          if np.ndim(freeflow_value) == 0
          else np.asarray(freeflow_value, dtype=float))

    for j in range(filled.shape[1]):
        col       = filled[:, j]
        valid_idx = np.where(~np.isnan(col))[0]
        if len(valid_idx) == 0:
            continue
        last_valid = valid_idx[-1]
        if last_valid + 1 < len(col) and not np.isnan(fv[j]):
            filled[last_valid + 1:, j] = fv[j]

    return filled


def fill_speed_nans(profile, minor_threshold=0.30, nodata_threshold=0.90):
    """
    Tiered NaN-handling policy for a speed (or speed-like) profile.

    NaN means "no vehicle observed on this segment at this time", NOT zero speed.
    Replacing NaN with 0 would falsely label the segment as fully congested.

    Cases per segment (column j in a (T, N) profile):
      - Case A  (nan_frac ≤ minor_threshold):   fill NaNs with the segment's temporal median.
      - Case C  (nan_frac > minor_threshold):    leave as NaN and flag in `nodata_mask` so
                                                  downstream visualisation renders the segment
                                                  in NODATA_COLOR (gray).
      - Case B  (neighbour fill, future work):   currently merged into Case C.

    Parameters
    ----------
    profile          : ndarray, shape (T, N) or (N,)
    minor_threshold  : float – max NaN fraction per segment for local median fill
    nodata_threshold : float – reserved for future Case B / C distinction

    Returns
    -------
    filled      : ndarray, same shape as `profile`
    nodata_mask : bool ndarray, shape (N,) – True for no-data segments
    """
    profile = np.asarray(profile, dtype=float)

    if profile.ndim == 1:
        nodata_mask = np.isnan(profile)
        return profile.copy(), nodata_mask

    filled = profile.copy()
    nan_frac    = np.mean(np.isnan(filled), axis=0)  # (N,)
    fillable    = nan_frac <= minor_threshold          # Case A
    nodata_mask = ~fillable                            # Cases B + C (gray for now)

    fill_cols = np.where(fillable)[0]
    if fill_cols.size:
        col_medians = np.nanmedian(filled[:, fill_cols], axis=0)
        # nanmedian returns NaN only if a "fillable" column is fully NaN, which contradicts
        # frac ≤ minor_threshold unless T is 0. Guard anyway to avoid propagating NaNs.
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        for i, c in enumerate(fill_cols):
            nan_rows = np.isnan(filled[:, c])
            filled[nan_rows, c] = col_medians[i]

    return filled, nodata_mask


def rowwise_zscore(profile):
    """Row-wise z-score normalisation of `profile`."""
    mean = profile.mean(axis=1, keepdims=True)
    std  = profile.std(axis=1,  keepdims=True)
    std[std == 0] = 1.0
    return (profile - mean) / std


def profile_components(profile, n_components=3):
    """Return the leading `n_components` SVD principal components of `profile`."""
    centered = profile - profile.mean(axis=0, keepdims=True)
    u, s, _  = np.linalg.svd(centered, full_matrices=False)
    n_comp   = min(n_components, centered.shape[0], centered.shape[1])
    return u[:, :n_comp] * s[:n_comp]
