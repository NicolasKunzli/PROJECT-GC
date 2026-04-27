"""
utils.py — Small standalone utilities used across the analysis pipeline.
"""

import numpy as np

from config import links


def closest_link(x, y, subset_idx=None):
    """
    Return the index of the link whose centre-point is nearest to (x, y).

    Uses the Euclidean (L2) norm on the pre-computed c_x / c_y columns.

    Parameters
    ----------
    x, y : float – query coordinates [m]
    subset_idx : array-like or None - Optional list/array of link indices to restrict the search space. If None, all links are considered.
    """
    if subset_idx is None:
        coords = np.column_stack([links["c_x"], links["c_y"]])
        idxs = np.arange(len(links))
    else:
        sub = links.iloc[subset_idx]
        coords = np.column_stack([sub["c_x"], sub["c_y"]])
        idxs = subset_idx

    distances = np.linalg.norm(coords - np.array([x, y]), axis=1)
    return idxs[np.argmin(distances)]