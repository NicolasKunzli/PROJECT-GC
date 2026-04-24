"""
network/simplify.py — Segment grouping and simplified network map.

Groups parallel/duplicate road segments (e.g. opposing lanes of the same road)
into a single representative, then renders a cleaner map coloured by average speed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

from config import DL, links, LOCAL_FIGURE, NODATA_COLOR
from network.draw import sublink, polyg
from processing.speed import mean_over_sessions


class UnionFind:
    """Path-compressed, union-by-rank disjoint-set forest."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def group_segments(distance_threshold=35.0, lateral_threshold=15.0):
    """
    Group road segments that represent the same physical road.

    Three matching criteria — any one is sufficient to merge two segments:
      1. Forward match  : both endpoint pairs are within `distance_threshold`
      2. Reverse match  : endpoints cross-match (opposite travel direction)
      3. Lateral match  : midpoint-to-midpoint distance ≤ `lateral_threshold`
                          (handles segments of different lengths on the same road)

    Parameters
    ----------
    distance_threshold : float – endpoint proximity threshold [m]
    lateral_threshold  : float – max midpoint distance for lateral pairing [m]

    Returns
    -------
    list of lists – each inner list contains the integer indices of one group
    """
    N       = len(links)
    from_xy = links[["from_x", "from_y"]].to_numpy()
    to_xy   = links[["to_x",   "to_y"  ]].to_numpy()

    from_tree = cKDTree(from_xy)
    to_tree   = cKDTree(to_xy)

    # 1. Forward match: from_i ≈ from_j  AND  to_i ≈ to_j
    forward_matches = from_tree.query_pairs(distance_threshold) & \
                      to_tree.query_pairs(distance_threshold)

    # 2. Reverse match: from_i ≈ to_j  AND  to_i ≈ from_j
    from_to_neighbors = from_tree.query_ball_tree(to_tree,   distance_threshold)
    to_from_neighbors = to_tree.query_ball_tree(from_tree, distance_threshold)

    reverse_matches = set()
    for i in range(N):
        candidates = set(from_to_neighbors[i]) & set(to_from_neighbors[i])
        for j in candidates:
            if i != j:
                reverse_matches.add((min(i, j), max(i, j)))

    # 3. Lateral match: midpoints within lateral_threshold
    mid_xy      = (from_xy + to_xy) / 2.0
    lateral_matches = cKDTree(mid_xy).query_pairs(lateral_threshold)

    all_matches = forward_matches | reverse_matches | lateral_matches

    uf = UnionFind(N)
    for i, j in all_matches:
        uf.union(i, j)

    groups_dict = {}
    for i in range(N):
        groups_dict.setdefault(uf.find(i), []).append(i)

    return list(groups_dict.values())


def compute_group_speeds(groups):
    """
    Mean speed [m/s] for each segment group (nanmean over all sessions, timesteps, members).

    Returns
    -------
    ndarray, shape (len(groups),)
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist, vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=vtime != 0,
    )

    group_speeds = np.empty(len(groups))
    for k, group in enumerate(groups):
        group_speeds[k] = np.nanmean(speed[:, :, group])

    return group_speeds


def simplified_map(distance_threshold, grad=True, color="navy"):
    """
    Render a simplified road network by keeping one representative segment per group.

    The representative is the segment with the most lanes (highest road hierarchy).

    Parameters
    ----------
    distance_threshold : float – passed to `group_segments`
    grad               : bool  – colour by average speed if True; flat colour otherwise
    color              : str   – flat colour used when grad=False
    """
    groups          = group_segments(distance_threshold)
    representatives = [max(g, key=lambda idx: links.iloc[idx]["num_lanes"]) for g in groups]

    fig, ax = plt.subplots(dpi=250)

    if grad:
        speeds       = compute_group_speeds(groups)
        valid_speeds = speeds[~np.isnan(speeds)]
        norm         = mcolors.Normalize(vmin=np.nanmin(valid_speeds), vmax=np.nanmax(valid_speeds))
        cmap         = plt.get_cmap("RdYlGn")

        for k, rep_idx in enumerate(representatives):
            row  = links.iloc[rep_idx]
            x, y = sublink(row)
            c    = NODATA_COLOR if np.isnan(speeds[k]) else cmap(norm(speeds[k]))
            ax.plot(x, y, c=c, linewidth=0.5)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Average speed [m/s]")
        suffix = "speed"
    else:
        for rep_idx in representatives:
            row  = links.iloc[rep_idx]
            x, y = sublink(row)
            ax.plot(x, y, c=color, linewidth=0.3)
        suffix = "flat"

    polyg(ax, color="black", alpha=0.3, zorder=-1)
    ax.set_aspect("equal")
    ax.set_title(
        f"Simplified network (threshold={distance_threshold}m, "
        f"{len(representatives)}/{len(links)} segments)",
        fontsize=9,
    )
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    out = f"{LOCAL_FIGURE}/simplified_map_{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved simplified map ({suffix}): {len(groups)} groups from {len(links)} segments")
