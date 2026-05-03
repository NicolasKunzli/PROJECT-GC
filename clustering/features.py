"""
clustering/features.py — Feature matrix builders and segment threshold detection.

All functions return a numpy array X of shape (n_links, n_features) ready for
sklearn clustering estimators. No plotting is done here.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import DL, links
from processing.speed import (
    mean_over_sessions,
    fill_speed_nans,
    rowwise_zscore,
    profile_components,
)


def thresholds(max_speed=0, max_length=0, session_min=0, session_max=100):
    """
    Return indices of links whose 85th-percentile speed ≤ max_speed AND length ≤ max_length.

    Parameters
    ----------
    max_speed  : float – 85th-pctile speed ceiling [m/s]; np.inf means no speed filter
    max_length : float – length ceiling [m]; np.inf means no length filter
    session_min, session_max : int – session slice passed to mean_over_sessions, session_max is included
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist, vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=vtime != 0,
    )

    speed_profile, nodata_mask = fill_speed_nans(
        mean_over_sessions(speed, min=session_min, max=session_max)
    )
    # nanpercentile returns NaN for no-data segments; exclude them since their real speed is unknown.
    speed_85 = np.nanpercentile(speed_profile, 85, axis=0)

    low_speed_links = np.where((speed_85 <= max_speed) & ~nodata_mask)[0]
    smallslow = [lnk for lnk in low_speed_links if links.iloc[lnk]["length"] <= max_length]

    return np.array(smallslow)


def temporal_cluster_features(
    profile,
    peak_mode,
    spatial_weight=2.5,
    dynamic_weight=1,
    threshold=np.array([]),
    filter=False,
):
    """
    Combined temporal + spatial feature matrix for segment clustering.

    Extracts per-segment: mean, std, peak timing, and 8 SVD components of the
    row-wise z-scored profile. Concatenates scaled spatial (x, y) coordinates.

    Parameters
    ----------
    profile        : ndarray, shape (N, T) – one row per segment, one col per timestep
    peak_mode      : "min" or "max" – direction of the congestion peak
    spatial_weight : float – scaling factor for spatial features after StandardScaler
    dynamic_weight : float – scaling factor for temporal features after StandardScaler
    threshold      : int ndarray – segment indices to remove when filter=True
    filter         : bool – drop threshold segments before building spatial features
    """
    filled, nodata_mask = fill_speed_nans(profile)

    # Case C segments still carry NaN; substitute global mean as a neutral placeholder
    # so clustering math (argmin / mean / std / SVD) can proceed. nodata_mask lets
    # downstream visualisation gray these segments out.
    if nodata_mask.any():
        neutral = np.nanmean(filled)
        if np.isnan(neutral):
            neutral = 0.0
        filled[np.isnan(filled)] = neutral

    peak_idx  = np.argmin(filled, axis=1) if peak_mode == "min" else np.argmax(filled, axis=1)
    peak_time = peak_idx / max(filled.shape[1] - 1, 1)

    dynamic = np.column_stack([
        filled.mean(axis=1),
        filled.std(axis=1),
        peak_time,
        profile_components(rowwise_zscore(filled), n_components=8),
    ])

    spatial_features = np.column_stack([links["c_x"].to_numpy(), links["c_y"].to_numpy()])

    if filter:
        threshold = threshold.astype(int)
        mask = np.ones(spatial_features.shape[0], dtype=bool)
        mask[threshold] = False
        spatial_features = spatial_features[mask, :]

    scaler = StandardScaler()
    dynamic_scaled = scaler.fit_transform(dynamic)  * dynamic_weight
    spatial_scaled = scaler.fit_transform(spatial_features) * spatial_weight

    return np.hstack([dynamic_scaled, spatial_scaled])


def build_cluster_features(
    feature_type,
    timeframe,
    spatial_weight=2.5,
    dynamic_weight=1,
    threshold=np.array([]),
    filter=False,
    one_timeframe=False,
    session_min=0,
    session_max=100,
):
    """
    Build a feature matrix X for clustering, keyed by `feature_type`.

    Parameters
    ----------
    feature_type  : {"geometric", "speed", "distance", "time"}
    timeframe     : int – timestep index used only when one_timeframe=True
    spatial_weight, dynamic_weight : float – passed to temporal_cluster_features
    threshold     : int ndarray – low-speed/short-link indices for optional filtering
    filter        : bool – drop threshold segments from the feature matrix
    one_timeframe : bool – build a snapshot (single-timestep) feature instead of temporal
    session_min, session_max : int – session slice for mean_over_sessions, session_max is included
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    if feature_type == "geometric":
        geometric = np.column_stack([
            DL.node_coordinates[:, 0],
            DL.node_coordinates[:, 1],
            links["length"].to_numpy(dtype=float),
            links["num_lanes"].to_numpy(dtype=float),
        ])
        return StandardScaler().fit_transform(geometric)

    if feature_type == "speed":
        speed = np.divide(
            vdist, vtime,
            out=np.full(vdist.shape, np.nan, dtype=float),
            where=vtime != 0,
        )
        speed_profile, nodata_mask = fill_speed_nans(
            mean_over_sessions(speed, min=session_min, max=session_max)
        )
        
        if filter and threshold.size > 0:
            mask = np.ones(speed_profile.shape[1], dtype=bool)
            mask[threshold] = False
            speed_profile = speed_profile[:, mask]
            nodata_mask   = nodata_mask[mask]

        if one_timeframe:
            # Snapshot mode: single timestep, spatial + speed features only.
            speed_snapshot = speed_profile[timeframe, :].copy()
            neutral = np.nanmean(speed_snapshot)
            if np.isnan(neutral):
                neutral = 0.0
            speed_snapshot = np.where(np.isnan(speed_snapshot), neutral, speed_snapshot)

            spatial = np.column_stack([links["c_x"].to_numpy(), links["c_y"].to_numpy()])

            if filter and threshold.size > 0:
                mask = np.ones(len(speed_snapshot), dtype=bool)
                mask[threshold] = False
                speed_snapshot = speed_snapshot[mask]
                spatial        = spatial[mask, :]

            X = StandardScaler().fit_transform(np.column_stack([speed_snapshot, spatial]))
            X[:, 0]  *= dynamic_weight
            X[:, 1:] *= spatial_weight
            return X

        return temporal_cluster_features(
            speed_profile.T, peak_mode="min",
            threshold=threshold, filter=filter,
        )

    if feature_type == "distance":
        distance = np.where(vdist != 0, vdist, np.nan)
        distance_profile = np.log1p(mean_over_sessions(distance), min=session_min, max=session_max).T
        return temporal_cluster_features(distance_profile, peak_mode="max")

    if feature_type == "time":
        time_profile = np.log1p(mean_over_sessions(np.where(vtime != 0, vtime, np.nan)), min=session_min, max=session_max).T
        return temporal_cluster_features(time_profile, peak_mode="max")

    raise ValueError(f"Unknown feature_type: {feature_type!r}")
