"""
percolation/graph_search.py — BFS connected-component analysis on the 20×16
congestion grid.

grid_graph_search_clustering : single-timestep BFS (8-connectivity) on congested
                               grid cells.
run_graph_search_analysis    : multi-timestep analysis + merged 3-row figure
                               (maps | cluster sizes | median speeds).
"""

import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import DL, links, LOCAL_FIGURE, NODATA_COLOR
from network.draw import polyg
from percolation.core import compute_normalized_speed
from processing.speed import fill_speed_nans, mean_over_sessions


# ── Constants ────────────────────────────────────────────────────────────────
XDIV = 20
YDIV = 16
N_TOP = 5   # number of top-ranked clusters to track

# rank-1 = red, rank-2 = blue, rank-3 = orange, rank-4 = purple, rank-5 = cyan
CLUSTER_PALETTE = ["#e6194b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
FUNCTIONAL_COLOR = "#2ca02c"


def _network_bounds(tol=100):
    x_min = min(float(np.min(links["from_x"])), float(np.min(links["to_x"]))) - tol
    x_max = max(float(np.max(links["from_x"])), float(np.max(links["to_x"]))) + tol
    y_min = min(float(np.min(links["from_y"])), float(np.min(links["to_y"]))) - tol
    y_max = max(float(np.max(links["from_y"])), float(np.max(links["to_y"]))) + tol
    return x_min, x_max, y_min, y_max


def _build_cell_map(speed_mean, nodata_mask, xdiv=XDIV, ydiv=YDIV):
    """
    Assign each link to a grid cell; compute per-cell average speed.

    Returns
    -------
    cell_avg      : ndarray (ydiv, xdiv) – mean speed per cell, NaN if empty/nodata
    cell_nodata   : bool ndarray (ydiv, xdiv) – True if cell has no valid links
    cell_link_map : dict (xi, yi) → list of link indices
    """
    x_min, x_max, y_min, y_max = _network_bounds()
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    cell_sum = np.zeros((ydiv, xdiv))
    cell_cnt = np.zeros((ydiv, xdiv), dtype=int)
    cell_link_map = {}

    for i, row in links.iterrows():
        xi = int(np.clip((row["c_x"] - x_min) / w, 0, xdiv - 1))
        yi = int(np.clip((row["c_y"] - y_min) / h, 0, ydiv - 1))
        key = (xi, yi)
        cell_link_map.setdefault(key, []).append(i)
        if not nodata_mask[i] and not np.isnan(speed_mean[i]):
            cell_sum[yi, xi] += speed_mean[i]
            cell_cnt[yi, xi] += 1

    cell_avg = np.full((ydiv, xdiv), np.nan)
    cell_nodata = np.ones((ydiv, xdiv), dtype=bool)
    for yi in range(ydiv):
        for xi in range(xdiv):
            if cell_cnt[yi, xi] > 0:
                cell_avg[yi, xi] = cell_sum[yi, xi] / cell_cnt[yi, xi]
                cell_nodata[yi, xi] = False

    return cell_avg, cell_nodata, cell_link_map


def _bfs_8conn(congested_grid):
    """
    BFS with 8-connectivity on a boolean grid.

    Returns
    -------
    cluster_map : int ndarray (ydiv, xdiv)
                  -1 = not congested / no-data, ≥0 = cluster id (0-indexed)
    n_clusters  : int
    """
    ydiv, xdiv = congested_grid.shape
    cluster_map = np.full((ydiv, xdiv), -1, dtype=int)
    visited = np.zeros((ydiv, xdiv), dtype=bool)
    cid = 0

    for yi in range(ydiv):
        for xi in range(xdiv):
            if congested_grid[yi, xi] and not visited[yi, xi]:
                q = deque([(yi, xi)])
                visited[yi, xi] = True
                while q:
                    cy, cx = q.popleft()
                    cluster_map[cy, cx] = cid
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < ydiv and 0 <= nx < xdiv
                                    and congested_grid[ny, nx]
                                    and not visited[ny, nx]):
                                visited[ny, nx] = True
                                q.append((ny, nx))
                cid += 1

    return cluster_map, cid


def grid_graph_search_clustering(speed_mean, nodata_mask, threshold,
                                 xdiv=XDIV, ydiv=YDIV):
    """
    BFS grid clustering at a single timestep.

    Parameters
    ----------
    speed_mean  : ndarray (N,) – per-segment speed value
    nodata_mask : bool ndarray (N,) – True if no data
    threshold   : float – congestion threshold (cells with avg speed < threshold
                  are congested)

    Returns
    -------
    cluster_map  : int ndarray (ydiv, xdiv) – BFS cluster labels
    n_clusters   : int – total number of congested BFS components
    cell_avg     : ndarray (ydiv, xdiv) – cell mean speed
    cell_nodata  : bool ndarray (ydiv, xdiv)
    cell_link_map: dict (xi, yi) → list of link indices
    """
    cell_avg, cell_nodata, cell_link_map = _build_cell_map(
        speed_mean, nodata_mask, xdiv, ydiv)
    congested = (~cell_nodata) & (cell_avg < threshold)
    cluster_map, n_clusters = _bfs_8conn(congested)
    return cluster_map, n_clusters, cell_avg, cell_nodata, cell_link_map


def _cluster_metrics(cluster_map, cell_link_map, raw_speed_t, n_top=N_TOP):
    """
    Compute size and median raw speed for the top-N clusters and the background.

    Returns
    -------
    ranked  : list of (cluster_id, size) sorted descending by size, length ≤ n_top
    sizes   : ndarray (n_top,) – sizes (NaN if cluster doesn't exist at this rank)
    medians : ndarray (n_top,) – median raw speed (NaN if absent)
    bg_size   : int – number of functional (background) cells
    bg_median : float – median raw speed of background links
    """
    # Count cell sizes per cluster
    cluster_sizes = {}
    for yi in range(cluster_map.shape[0]):
        for xi in range(cluster_map.shape[1]):
            cid = cluster_map[yi, xi]
            if cid >= 0:
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

    ranked = sorted(cluster_sizes.items(), key=lambda x: -x[1])[:n_top]

    sizes   = np.full(n_top, np.nan)
    medians = np.full(n_top, np.nan)

    for rank, (cid, sz) in enumerate(ranked):
        sizes[rank] = sz
        link_speeds = []
        for yi in range(cluster_map.shape[0]):
            for xi in range(cluster_map.shape[1]):
                if cluster_map[yi, xi] == cid:
                    for li in cell_link_map.get((xi, yi), []):
                        v = raw_speed_t[li]
                        if not np.isnan(v):
                            link_speeds.append(v)
        if link_speeds:
            medians[rank] = np.median(link_speeds)

    # Background
    bg_links = []
    bg_size  = 0
    for yi in range(cluster_map.shape[0]):
        for xi in range(cluster_map.shape[1]):
            if cluster_map[yi, xi] == -1:
                bg_size += 1
                for li in cell_link_map.get((xi, yi), []):
                    v = raw_speed_t[li]
                    if not np.isnan(v):
                        bg_links.append(v)
    bg_median = float(np.median(bg_links)) if bg_links else np.nan

    return ranked, sizes, medians, bg_size, bg_median


def _draw_grid_map(ax, cluster_map, cell_nodata, ranked, xdiv=XDIV, ydiv=YDIV,
                   title=""):
    """Draw a single BFS-cluster grid map on axis `ax`."""
    x_min, x_max, y_min, y_max = _network_bounds()
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    rank_color = {cid: CLUSTER_PALETTE[r % len(CLUSTER_PALETTE)]
                  for r, (cid, _) in enumerate(ranked)}

    for yi in range(ydiv):
        for xi in range(xdiv):
            x = x_min + xi * w
            y = y_min + yi * h
            cid = cluster_map[yi, xi]
            if cell_nodata[yi, xi]:
                fc = NODATA_COLOR
            elif cid >= 0:
                fc = rank_color.get(cid, "#888888")
            else:
                fc = FUNCTIONAL_COLOR
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                edgecolor="black", facecolor=fc, alpha=0.55, linewidth=0.3,
            ))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("X", fontsize=6)
    ax.set_ylabel("Y", fontsize=6)


def run_graph_search_analysis(qc=0.52, mode="norm",
                              timesteps=None, raw_threshold=2.0,
                              map_timesteps=None):
    """
    Run BFS grid analysis over 15 representative timesteps and save a merged figure.

    Layout (3 rows):
      Row 1 – spatial cluster maps at 4 selected timesteps
      Row 2 – BFS cluster sizes (top-5) over time
      Row 3 – median raw speed (m/s) per top-5 cluster + background over time

    Parameters
    ----------
    qc            : float – normalised speed threshold (mode='norm')
    mode          : 'norm' (normalised) or 'raw' (raw m/s)
    timesteps     : list[int] or None – 15 standard report timesteps if None
    raw_threshold : float – raw-speed threshold in m/s (mode='raw' only)
    map_timesteps : list[int] or None – which 4 timesteps to show as maps

    Returns
    -------
    dict with keys: timesteps, n_clusters, sizes (T×5), medians (T×5), bg_median (T,)
    """
    if timesteps is None:
        timesteps = [0, 3, 5, 7, 9, 13, 17, 19, 21, 23, 25, 29, 33, 36, 38]
    if map_timesteps is None:
        # spread across window
        map_timesteps = [timesteps[0], timesteps[3], timesteps[8], timesteps[-1]]

    # ── Build per-timestep speed arrays (session-averaged + NaN-filled) ───────
    r = compute_normalized_speed()          # (S, T, N)
    r_mean = mean_over_sessions(r, min=0, max=r.shape[0])   # (T, N)
    r_profile, nodata_mask = fill_speed_nans(r_mean)        # (T, N), (N,)

    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)
    raw_all = np.divide(vdist, vtime,
                        out=np.full(vdist.shape, np.nan, dtype=float),
                        where=(vtime != 0) & (~np.isnan(vtime)))  # (S, T, N)
    raw_mean = mean_over_sessions(raw_all, min=0, max=raw_all.shape[0])  # (T, N)
    raw_profile, _ = fill_speed_nans(raw_mean)                           # (T, N)

    if mode == "norm":
        speed_profile = r_profile
        threshold     = qc
        folder_name   = f"graph_search/norm_q{qc:.2f}"
        title_prefix  = f"BFS — normalised speed, $q_c = {qc:.2f}$"
    else:
        speed_profile = raw_profile
        threshold     = raw_threshold
        folder_name   = f"graph_search/raw_q{raw_threshold:.0f}"
        title_prefix  = f"BFS — raw speed, $q = {raw_threshold:.1f}$\\,m/s"

    folder = os.path.join(LOCAL_FIGURE, folder_name)
    os.makedirs(folder, exist_ok=True)

    # ── Collect metrics ───────────────────────────────────────────────────────
    all_n_clusters = []
    all_sizes      = []
    all_medians    = []
    all_bg_median  = []
    stored_maps    = {}   # t → (cluster_map, cell_nodata, ranked)

    for t in timesteps:
        cluster_map, n_clusters, cell_avg, cell_nodata, cell_link_map = \
            grid_graph_search_clustering(speed_profile[t], nodata_mask, threshold)

        ranked, sizes, medians, bg_size, bg_median = \
            _cluster_metrics(cluster_map, cell_link_map, raw_profile[t])

        all_n_clusters.append(n_clusters)
        all_sizes.append(sizes)
        all_medians.append(medians)
        all_bg_median.append(bg_median)

        if t in map_timesteps:
            stored_maps[t] = (cluster_map, cell_nodata, ranked)

    all_sizes    = np.array(all_sizes)    # (T, N_TOP)
    all_medians  = np.array(all_medians)  # (T, N_TOP)
    all_bg_median = np.array(all_bg_median)

    # ── Build merged figure ───────────────────────────────────────────────────
    n_maps = len(map_timesteps)
    fig = plt.figure(figsize=(4 * n_maps, 11), dpi=150)

    # Row 1: spatial maps
    for col, t in enumerate(map_timesteps):
        ax = fig.add_subplot(3, n_maps, col + 1)
        cluster_map, cell_nodata, ranked = stored_maps[t]
        total_min = 8 * 60 + 3 + t * 3
        tstr = f"{total_min // 60:02d}:{total_min % 60:02d}"
        _draw_grid_map(ax, cluster_map, cell_nodata, ranked, title=tstr)

    # Row 2: cluster sizes
    ax_sz = fig.add_subplot(3, 1, 2)
    for rank in range(N_TOP):
        s = all_sizes[:, rank]
        ok = ~np.isnan(s)
        if ok.any():
            ax_sz.plot(np.array(timesteps)[ok], s[ok],
                       label=f"Cluster {rank + 1}",
                       color=CLUSTER_PALETTE[rank],
                       marker="o", markersize=3, linewidth=1.2)
    for t in map_timesteps:
        ax_sz.axvline(t, color="gray", linestyle="--", linewidth=0.6)
    ax_sz.set_xlabel("Timestep index", fontsize=8)
    ax_sz.set_ylabel("# grid cells", fontsize=8)
    ax_sz.set_title("Congested cluster sizes (top-5 by size)", fontsize=8)
    ax_sz.legend(fontsize=7)
    ax_sz.tick_params(labelsize=7)

    # Row 3: median speed
    ax_med = fig.add_subplot(3, 1, 3)
    for rank in range(N_TOP):
        m = all_medians[:, rank]
        ok = ~np.isnan(m)
        if ok.any():
            ax_med.plot(np.array(timesteps)[ok], m[ok],
                        label=f"Cluster {rank + 1}",
                        color=CLUSTER_PALETTE[rank],
                        marker="o", markersize=3, linewidth=1.2)
    ok_bg = ~np.isnan(all_bg_median)
    ax_med.plot(np.array(timesteps)[ok_bg], all_bg_median[ok_bg],
                label="Background (functional)",
                color=FUNCTIONAL_COLOR,
                linestyle="--", marker="s", markersize=3, linewidth=1.2)
    for t in map_timesteps:
        ax_med.axvline(t, color="gray", linestyle="--", linewidth=0.6)
    ax_med.set_xlabel("Timestep index", fontsize=8)
    ax_med.set_ylabel("Median raw speed (m/s)", fontsize=8)
    ax_med.set_title("Median raw speed per cluster", fontsize=8)
    ax_med.legend(fontsize=7)
    ax_med.tick_params(labelsize=7)

    fig.suptitle(title_prefix + "\n(20×16 grid, 8-connectivity BFS)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(folder, "merged_graph_search_analysis.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    return {
        "timesteps":   timesteps,
        "n_clusters":  np.array(all_n_clusters),
        "sizes":       all_sizes,
        "medians":     all_medians,
        "bg_median":   all_bg_median,
    }
