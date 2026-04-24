"""
utils.py — Small standalone utilities used across the analysis pipeline.
"""

import numpy as np

from config import links


def closest_link(x, y):
    """
    Return the index of the link whose centre-point is nearest to (x, y).

    Uses the Euclidean (L2) norm on the pre-computed c_x / c_y columns.

    Parameters
    ----------
    x, y : float – query coordinates [m]
    """
    coords    = np.column_stack([links["c_x"].to_numpy(), links["c_y"].to_numpy()])
    distances = np.linalg.norm(coords - np.array([x, y]), axis=1)
    return np.argmin(distances)
