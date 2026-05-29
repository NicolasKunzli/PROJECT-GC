"""
percolation/graph_search.py — Grid-based graph-search clustering for congestion.

Algorithm (per timestep)
------------------------
1. Divide the network into an xdiv × ydiv grid.
2. Compute the mean normalised speed r̄ for every cell (links assigned by centroid).
3. Label each cell:  congested (r̄ < qc) | functional (r̄ ≥ qc) | no-data (empty).
4. Build a grid graph where two cells are neighbours iff they share an EDGE or a CORNER
   (8-connectivity: up/down/left/right + all 4 diagonals).
   Rationale: each grid cell spans ~200–300 m of real city; diagonal cells almost
   always share road infrastructure, so excluding them artificially fragments
   congestion zones that are physically contiguous.
5. Find connected components with explicit BFS visiting all 8 neighbours.
6. Track the top-5 component sizes over all timesteps.

Outputs
-------
• figure/graph_search/<folder>/graph_search_t<t>.png        — map at one timestep
• figure/graph_search/<folder>/top5_cluster_sizes.png       — time-series of top-5 sizes
• figure/graph_search/<folder>/median_speeds.png            — median raw speed per cluster
• figure/graph_search/<folder>/merged_graph_search_analysis.png — 3-row combined figure
"""

import math
import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import DL, links, LOCAL_FIGURE, NODATA_COLOR
from network.draw import sublink
from network.simplify import group_segments
from percolation.core import compute_normalized_speed
from processing.speed import fill_speed_nans, fill_trailing_nans_as_freeflow, mean_over_sessions


# ── Constants ──────────────────────────────────────────────────────────────────

# Colours for 5 congested clusters (rank 0 = largest)
_TOP5_COLORS    = ["#e6194b", "#4363d8", "#f58231", "#3cb44b", "#911eb4"]
_SMALL_CONG_COLOR = "#ffbbbb"   # faded pink for smaller congested clusters
_FUNC_COLOR       = "#33aa33"   # green for functional segments

# 8-neighbours: cardinal directions + all 4 diagonals.
# At the 20×16 grid resolution (~200–300 m per cell) diagonal cells share
# real road infrastructure, so 8-connectivity gives a more physically
# meaningful definition of a "contiguous congestion zone".
_NEIGHBOURS_8 = (
    (-1,  0), (1,  0), (0, -1), (0,  1),   # cardinal
    (-1, -1), (-1,  1), (1, -1), (1,  1),  # diagonal
)

# Dataset timestamps (3-min intervals starting at 08:03)
_TS_BASE = pd.date_range("2005-05-10 08:03", periods=200, freq="3min")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _network_bounds(tol=100):
    x_min = min(links["from_x"].min(), links["to_x"].min()) - tol
    x_max = max(links["from_x"].max(), links["to_x"].max()) + tol
    y_min = min(links["from_y"].min(), links["to_y"].min()) - tol
    y_max = max(links["from_y"].max(), links["to_y"].max()) + tol
    return x_min, x_max, y_min, y_max


def _links_with_cells(bounds, xdiv, ydiv):
    """Return a copy of `links` with 'cell_x' and 'cell_y' columns pre-computed."""
    x_min, x_max, y_min, y_max = bounds
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    df = links[["c_x", "c_y"]].copy()
    df["cell_x"] = np.clip(((df["c_x"] - x_min) // w).astype(int), 0, xdiv - 1)
    df["cell_y"] = np.clip(((df["c_y"] - y_min) // h).astype(int), 0, ydiv - 1)
    return df


def _assign_cell_state(r_t, nodata_mask, links_cells, xdiv, ydiv, qc,
                       cong_frac_threshold=0.40, nan_at_t=None):
    """
    Classify each grid cell as congested, functional, or no-data.

    A cell is **congested** if the fraction of its segments with a measured
    r < qc exceeds `cong_frac_threshold` (default 40 %).

    Segments without data at this specific timestep (`nan_at_t=True`) are
    included in the denominator but treated as uncongested — no observation
    means no detected congestion.  Global no-data segments (`nodata_mask=True`)
    are excluded entirely (too many missing observations to be meaningful).
    Cells with no present segments remain -1 (no-data).

    Parameters
    ----------
    nan_at_t : bool ndarray (N,) or None
        Per-segment NaN flags BEFORE any filling, for this specific timestep.
        When provided, NaN-at-t segments count as uncongested in the cell fraction.

    Returns
    -------
    grid_state : ndarray (xdiv, ydiv), values  -1 = no-data, 0 = functional, 1 = congested
    """
    # Present = not a globally-excluded no-data segment
    present = ~nodata_mask

    # Congested = present + had valid data at t + r < qc
    # NaN at this timestep → uncongested (in denominator, not numerator)
    if nan_at_t is not None:
        is_congested = present & ~nan_at_t & (r_t < qc)
    else:
        is_congested = present & ~np.isnan(r_t) & (r_t < qc)

    df = pd.DataFrame({
        "cell_x":    links_cells["cell_x"].values,
        "cell_y":    links_cells["cell_y"].values,
        "congested": is_congested.astype(int),
        "valid":     present,
    })

    df_valid = df[df["valid"]]
    cell_counts = df_valid.groupby(["cell_x", "cell_y"])["congested"].agg(
        total="count", n_cong="sum"
    )

    grid_state = np.full((xdiv, ydiv), -1, dtype=int)
    for (cx, cy), row in cell_counts.iterrows():
        cx, cy = int(cx), int(cy)
        if 0 <= cx < xdiv and 0 <= cy < ydiv:
            frac = row["n_cong"] / row["total"]
            grid_state[cx, cy] = 1 if frac > cong_frac_threshold else 0

    return grid_state


def _bfs_labels(binary_grid):
    """
    BFS connected-components with 8-connectivity (cardinal + diagonal neighbours).

    Parameters
    ----------
    binary_grid : bool ndarray (rows, cols)

    Returns
    -------
    labels       : int ndarray (rows, cols) — 0 = background, 1..n = component id
    n_components : int
    """
    rows, cols = binary_grid.shape
    labels = np.zeros((rows, cols), dtype=int)
    n_comp = 0

    for r in range(rows):
        for c in range(cols):
            if binary_grid[r, c] and labels[r, c] == 0:
                n_comp += 1
                labels[r, c] = n_comp
                queue = deque([(r, c)])
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in _NEIGHBOURS_8:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and binary_grid[nr, nc]
                                and labels[nr, nc] == 0):
                            labels[nr, nc] = n_comp
                            queue.append((nr, nc))

    return labels, n_comp


def _find_components(grid_state):
    """
    Find connected components for congested (1) and functional (0) cells
    using BFS with 8-connectivity (cardinal + diagonal neighbours).

    Returns: (cong_labels, n_cong, func_labels, n_func)
    """
    cong_lab, n_cong = _bfs_labels(grid_state == 1)
    func_lab, n_func = _bfs_labels(grid_state == 0)
    return cong_lab, n_cong, func_lab, n_func


def _sorted_sizes(labels, n_comp):
    """Component sizes sorted descending (excluding background label 0)."""
    if n_comp == 0:
        return []
    return sorted(
        [int((labels == i).sum()) for i in range(1, n_comp + 1)],
        reverse=True,
    )


def _build_rank_maps(cong_labels_all):
    """
    Per-timestep size ranking: at each timestep rank every congested BFS component
    by the number of cells it contains (rank 1 = largest).

    Returns
    -------
    rank_maps : list[dict[int, int]]  — rank_maps[t][bfs_label] = rank (1-based)
    """
    rank_maps = []
    for cl in cong_labels_all:
        nc = int(cl.max())
        if nc == 0:
            rank_maps.append({})
            continue
        sizes      = {i: int((cl == i).sum()) for i in range(1, nc + 1)}
        sorted_ids = sorted(sizes, key=lambda x: -sizes[x])
        rank_maps.append({comp_id: rank + 1 for rank, comp_id in enumerate(sorted_ids)})
    return rank_maps


def _compute_ranked_sizes(results, top_n=5):
    """
    Size (# grid cells) of the top-N ranked congested clusters at every timestep.

    Rank is determined per-timestep by component size: rank 1 = largest cluster
    at that timestep.

    Returns ndarray (T, top_n) — 0 when the rank-k cluster is absent.
    """
    T         = results["T"]
    rank_maps = results["rank_maps"]
    sizes     = np.zeros((T, top_n), dtype=int)

    for t in range(T):
        cl       = results["cong_labels"][t]
        for bfs_label, rank in rank_maps[t].items():
            if rank <= top_n:
                sizes[t, rank - 1] = int((cl == bfs_label).sum())

    return sizes


def _compute_median_speeds(results, top_n=5, min_cells=1):
    """
    Median raw speed (m/s) for the top-N persistent congested clusters at every
    timestep.  Cluster identity is tracked spatially across time (persistent IDs),
    so the same colour always refers to the same geographic cluster.

    A quality filter removes isolated single-timestep points: if a cluster has valid
    data at time t but NaN at both t-1 and t+1, the value is removed.

    Parameters
    ----------
    results   : dict returned by grid_graph_search_clustering
    top_n     : int – how many size-ranked clusters to include (≤ 5)
    min_cells : int – minimum cluster size (grid cells) to include

    Returns
    -------
    ndarray (T, top_n) – NaN when cluster is absent or isolated
    """
    T           = results["T"]
    raw_profile = results["raw_speed_profile"]   # (T, N)  m/s
    lc          = results["links_cells"]
    rank_maps   = results["rank_maps"]

    lx = lc["cell_x"].values.astype(int)    # (N,)
    ly = lc["cell_y"].values.astype(int)    # (N,)

    median_speeds = np.full((T, top_n), np.nan)

    for t in range(T):
        cl          = results["cong_labels"][t]   # (xdiv, ydiv)
        link_labels = cl[lx, ly]                  # (N,)

        for bfs_label, rank in rank_maps[t].items():
            if rank > top_n:
                continue
            if int((cl == bfs_label).sum()) < min_cells:
                continue
            mask  = link_labels == bfs_label
            valid = raw_profile[t, mask]
            valid = valid[~np.isnan(valid)]
            if len(valid):
                median_speeds[t, rank - 1] = np.median(valid)

    # Remove isolated single-timestep points (artefacts of cluster disappearance)
    for k in range(top_n):
        for t in range(T):
            prev_nan = np.isnan(median_speeds[t - 1, k]) if t > 0     else True
            next_nan = np.isnan(median_speeds[t + 1, k]) if t < T - 1 else True
            if not np.isnan(median_speeds[t, k]) and prev_nan and next_nan:
                median_speeds[t, k] = np.nan

    return median_speeds


def _simplified_rep_links(bounds, xdiv, ydiv, distance_threshold=50.0):
    """
    Build representative-link DataFrame for the simplified network,
    annotated with grid-cell assignments.
    """
    x_min, x_max, y_min, y_max = bounds
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    groups = group_segments(distance_threshold)
    reps   = [max(g, key=lambda idx: links.iloc[idx]["num_lanes"]) for g in groups]
    rep_df = links.iloc[reps].copy().reset_index(drop=True)

    rep_df["cell_x"] = np.clip(
        ((rep_df["c_x"] - x_min) // w).astype(int), 0, xdiv - 1
    )
    rep_df["cell_y"] = np.clip(
        ((rep_df["c_y"] - y_min) // h).astype(int), 0, ydiv - 1
    )
    return rep_df


# ── Main analysis function ─────────────────────────────────────────────────────

def grid_graph_search_clustering(
    qc=0.52,
    xdiv=20,
    ydiv=16,
    session=0,
    use_all_sessions=True,
    use_normalized=True,
):
    """
    Run graph-search clustering on the grid for every timestep.

    Parameters
    ----------
    qc              : float – congestion threshold.
                      If use_normalized=True : normalised speed threshold r̄_cell < qc.
                      If use_normalized=False: raw speed threshold in m/s (v̄_cell < qc).
    xdiv, ydiv      : int   – grid divisions (columns, rows)
    session         : int   – session index used when use_all_sessions=False
    use_all_sessions: bool  – if True, average over all sessions before analysis
    use_normalized  : bool  – True  → classify cells by normalised speed r ∈ [0,1]
                              False → classify cells by raw speed (m/s)

    Returns
    -------
    dict with keys:
        'cong_sizes'       : list[list[int]]  – sorted cluster sizes (congested) per t
        'func_sizes'       : list[list[int]]  – sorted cluster sizes (functional) per t
        'grid_states'      : list[ndarray]    – (xdiv, ydiv) state grids
        'cong_labels'      : list[ndarray]    – BFS label arrays (congested components)
        'func_labels'      : list[ndarray]    – BFS label arrays (functional components)
        'T'                : int
        'qc'               : float
        'use_normalized'   : bool
        'xdiv','ydiv'      : int
        'bounds'           : (x_min, x_max, y_min, y_max)
        'nodata_mask'      : bool ndarray (N,)
        'raw_speed_profile': ndarray (T, N)  – raw speed in m/s (for median computation)
        'links_cells'      : DataFrame       – per-link cell assignments (cell_x, cell_y)
    """
    # ── Always compute raw speed profile (needed for median speed subplot) ──
    vdist     = DL._vdist_3min.astype(float)
    vtime     = DL._vtime_3min.astype(float)
    speed_raw = np.divide(vdist, vtime,
                          out=np.full(vdist.shape, np.nan, dtype=float),
                          where=vtime != 0)

    # Per-segment free-flow proxy: 95th percentile over all sessions/timesteps.
    # Used to fill trailing NaNs in raw speed (end-of-simulation gaps → uncongested).
    v_max_raw = np.nanpercentile(speed_raw.reshape(-1, speed_raw.shape[2]), 95, axis=0)
    v_max_raw[v_max_raw == 0] = np.nan

    if use_all_sessions:
        # Fill trailing NaN per session BEFORE averaging to avoid end-of-sim selection bias:
        # sessions that end early would otherwise be absent from the mean at late timesteps,
        # leaving only the (potentially atypical) long-running sessions.
        S = speed_raw.shape[0]
        speed_raw_filled = np.empty_like(speed_raw)
        for s in range(S):
            speed_raw_filled[s] = fill_trailing_nans_as_freeflow(
                speed_raw[s], freeflow_value=v_max_raw
            )
        raw_mean = mean_over_sessions(speed_raw_filled, min=0, max=S)
        raw_nan  = np.isnan(raw_mean)                           # (T, N) before interior fill
        raw_profile, raw_nodata = fill_speed_nans(raw_mean)
    else:
        raw_s0   = speed_raw[session]
        raw_nan  = np.isnan(raw_s0)
        raw_s0   = fill_trailing_nans_as_freeflow(raw_s0, freeflow_value=v_max_raw)
        raw_profile, raw_nodata = fill_speed_nans(raw_s0)

    # ── Values used for cell-state classification ───────────────────────────
    if use_normalized:
        r = compute_normalized_speed()                           # (S, T, N)
        if use_all_sessions:
            S = r.shape[0]
            r_filled = np.empty_like(r)
            for s in range(S):
                r_filled[s] = fill_trailing_nans_as_freeflow(r[s], freeflow_value=1.0)
            r_mean     = mean_over_sessions(r_filled, min=0, max=S)
            values_nan = np.isnan(r_mean)                       # (T, N) before interior fill
            values, nodata_mask = fill_speed_nans(r_mean)
        else:
            r_s        = r[session]
            values_nan = np.isnan(r_s)
            r_s        = fill_trailing_nans_as_freeflow(r_s, freeflow_value=1.0)
            values, nodata_mask = fill_speed_nans(r_s)
    else:
        values      = raw_profile
        nodata_mask = raw_nodata
        values_nan  = raw_nan

    T      = values.shape[0]
    bounds = _network_bounds()
    lc     = _links_with_cells(bounds, xdiv, ydiv)

    cong_sizes_all, func_sizes_all = [], []
    grid_states, cong_labels_all, func_labels_all = [], [], []

    mode = "normalised r" if use_normalized else "raw speed m/s"
    print(f"  Graph search: processing {T} timesteps  ({mode}, q={qc}) …")
    for t in range(T):
        gs             = _assign_cell_state(values[t], nodata_mask, lc, xdiv, ydiv, qc,
                                            nan_at_t=values_nan[t])
        cl, nc, fl, nf = _find_components(gs)

        cong_sizes_all.append(_sorted_sizes(cl, nc))
        func_sizes_all.append(_sorted_sizes(fl, nf))
        grid_states.append(gs)
        cong_labels_all.append(cl)
        func_labels_all.append(fl)

    print(f"  Done.  (q={qc}, {xdiv}×{ydiv} grid, {T} timesteps)")

    print("  Building per-timestep size rankings …")
    rank_maps = _build_rank_maps(cong_labels_all)

    return dict(
        cong_sizes=cong_sizes_all,
        func_sizes=func_sizes_all,
        grid_states=grid_states,
        cong_labels=cong_labels_all,
        func_labels=func_labels_all,
        T=T,
        qc=qc,
        use_normalized=use_normalized,
        xdiv=xdiv,
        ydiv=ydiv,
        bounds=bounds,
        nodata_mask=nodata_mask,
        raw_speed_profile=raw_profile,   # always raw m/s, for median plot
        links_cells=lc,                  # cell assignments per link
        rank_maps=rank_maps,             # per-timestep size ranking
    )


# ── Time-series plots ──────────────────────────────────────────────────────────

def plot_top5_cluster_sizes(results, folder, fname="top5_cluster_sizes.png"):
    """
    Standalone full-size line chart: top-5 congested cluster sizes over time.
    """
    T    = results["T"]
    qc   = results["qc"]
    xdiv = results["xdiv"]
    ydiv = results["ydiv"]

    top5      = _compute_ranked_sizes(results, top_n=5)   # (T, 5)
    ts_labels = [_TS_BASE[i].strftime("%H:%M") for i in range(T)]
    tick_step = max(1, T // 10)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

    for k in range(5):
        if top5[:, k].max() == 0:
            continue
        ax.plot(range(T), top5[:, k],
                color=_TOP5_COLORS[k],
                linewidth=1.8 if k == 0 else 1.2,
                marker="o", markersize=3,
                label=f"Cluster {k + 1}")

    ax.set_xticks(range(0, T, tick_step))
    ax.set_xticklabels(ts_labels[::tick_step], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Cluster size  (# grid cells)", fontsize=10)
    ax.set_title(
        f"Top-5 Congested Cluster Sizes Over Time  "
        f"(q = {qc:.2f},  {xdiv}×{ydiv} grid,  mean over all sessions)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved top-5 cluster sizes → {out}")
    return out


def plot_median_speeds(results, folder, fname="median_speeds.png"):
    """
    Standalone full-size line chart: median raw speed (m/s) of the top-5
    congested clusters over all timesteps.
    """
    T    = results["T"]
    qc   = results["qc"]
    xdiv = results["xdiv"]
    ydiv = results["ydiv"]

    median_spd = _compute_median_speeds(results, top_n=5)   # (T, 5)

    ts_labels = [_TS_BASE[i].strftime("%H:%M") for i in range(T)]
    tick_step = max(1, T // 10)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

    plotted = False
    for k in range(5):
        col_data = median_spd[:, k]
        if np.all(np.isnan(col_data)):
            continue
        ax.plot(range(T), col_data,
                color=_TOP5_COLORS[k],
                linewidth=2.0 if k == 0 else 1.4,
                marker="o", markersize=4,
                label=f"Cluster {k + 1}")
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No congested clusters above threshold",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")

    ax.set_xticks(range(0, T, tick_step))
    ax.set_xticklabels(ts_labels[::tick_step], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Median speed  (m/s)", fontsize=10)
    ax.set_title(
        f"Median Raw Speed of Top-5 Congested Clusters  "
        f"(q = {qc:.2f},  {xdiv}×{ydiv} grid,  mean over all sessions)",
        fontsize=11,
    )
    if plotted:
        ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved median speeds → {out}")
    return out


# ── Map drawing ────────────────────────────────────────────────────────────────

def _draw_cluster_map(ax, t, results, rep_links):
    """
    Draw the graph-search cluster map for timestep `t` on `ax`.

    Congested clusters are coloured by per-timestep size rank: the largest
    congested component at this timestep gets color 1, second-largest gets
    color 2, etc. (up to 5 distinct colors); all smaller clusters get faded pink.
    Functional cells are drawn in green.
    No-data cells/segments are drawn in NODATA_COLOR.
    """
    xdiv      = results["xdiv"]
    ydiv      = results["ydiv"]
    bounds    = results["bounds"]
    x_min, x_max, y_min, y_max = bounds
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    gs       = results["grid_states"][t]
    cl       = results["cong_labels"][t]
    rank_map = results["rank_maps"][t]     # bfs_label -> rank (1 = largest at this t)

    def _cong_color(comp_id):
        rank = rank_map.get(comp_id, 999)
        return _TOP5_COLORS[rank - 1] if rank <= 5 else _SMALL_CONG_COLOR

    # ── Draw grid cell rectangles (faded background) ───────────────────────
    for cx in range(xdiv):
        for cy in range(ydiv):
            state = gs[cx, cy]
            x0, y0 = x_min + cx * w, y_min + cy * h
            if state == 1:
                fc, alpha = _cong_color(cl[cx, cy]), 0.28
            elif state == 0:
                fc, alpha = "#b8ffb8", 0.18
            else:
                fc, alpha = "none", 0.0
            ax.add_patch(patches.Rectangle(
                (x0, y0), w, h,
                edgecolor="black", facecolor=fc, alpha=alpha,
                linewidth=0.3, zorder=1,
            ))

    # ── Draw simplified road segments coloured by cluster ─────────────────
    for _, row in rep_links.iterrows():
        cx    = int(np.clip(row["cell_x"], 0, xdiv - 1))
        cy    = int(np.clip(row["cell_y"], 0, ydiv - 1))
        state = gs[cx, cy]

        if state == -1:
            color, lw = NODATA_COLOR, 0.4
        elif state == 1:
            color, lw = _cong_color(cl[cx, cy]), 0.9
        else:
            color, lw = _FUNC_COLOR, 0.7

        xs, ys = sublink(row)
        ax.plot(xs, ys, c=color, linewidth=lw, zorder=2)

    ts_str = _TS_BASE[t].strftime("%H:%M")
    nc_c   = int((gs == 1).sum())
    nc_f   = int((gs == 0).sum())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(
        f"t={t}  ({ts_str})\n{nc_c} congested / {nc_f} functional cells",
        fontsize=7,
    )
    ax.tick_params(axis="both", labelsize=5)


def _cluster_legend_handles():
    handles = [
        plt.Line2D([0], [0], color=_TOP5_COLORS[k], lw=2,
                   label=f"Congested cluster {k + 1}")
        for k in range(5)
    ]
    handles += [
        plt.Line2D([0], [0], color=_SMALL_CONG_COLOR, lw=2, label="Congested (small)"),
        plt.Line2D([0], [0], color=_FUNC_COLOR,        lw=2, label="Functional"),
        plt.Line2D([0], [0], color=NODATA_COLOR,        lw=2, label="No data"),
    ]
    return handles


def plot_all_timestep_maps(results, folder, rep_links, fname="all_timestep_maps.png"):
    """
    Grid figure showing the congestion map at every timestep.

    Layout: ncols = ceil(sqrt(T)) columns, nrows = ceil(T / ncols) rows.
    Empty panels (last row) are hidden.
    """
    T     = results["T"]
    qc    = results["qc"]
    xdiv  = results["xdiv"]
    ydiv  = results["ydiv"]

    ncols = math.ceil(math.sqrt(T))
    nrows = math.ceil(T / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        dpi=150,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for t in range(T):
        r, c = divmod(t, ncols)
        _draw_cluster_map(axes[r, c], t, results, rep_links)

    # Legend in the first panel
    axes[0, 0].legend(
        handles=_cluster_legend_handles(), fontsize=4, loc="lower left", ncol=1
    )

    # Hide unused panels
    for t in range(T, nrows * ncols):
        r, c = divmod(t, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Graph-Search Clusters — All {T} Timesteps  "
        f"(q = {qc:.2f},  {xdiv}×{ydiv} grid)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved all-timestep map grid → {out}")
    return out


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_graph_search_analysis(
    qc=0.52,
    xdiv=20,
    ydiv=16,
    key_timesteps=None,
    session=0,
    use_all_sessions=True,
    use_normalized=True,
    folder=None,
):
    """
    Full graph-search clustering pipeline — 3-panel merged figure:

      Row 1 : maps at key_timesteps (coloured by cluster rank)
      Row 2 : top-5 congested cluster sizes over time (# grid cells)
      Row 3 : median raw speed (m/s) of the same top-5 clusters over time

    Parameters
    ----------
    qc              : float – congestion threshold (normalised or m/s)
    xdiv, ydiv      : int
    key_timesteps   : list[int] – timesteps shown on maps
    session         : int       – used when use_all_sessions=False
    use_all_sessions: bool
    use_normalized  : bool      – True → normalised speed; False → raw speed m/s
    folder          : str       – output directory

    Returns
    -------
    results dict from grid_graph_search_clustering
    """
    if folder is None:
        folder = os.path.join(LOCAL_FIGURE, "graph_search")
    os.makedirs(folder, exist_ok=True)

    # ── 1. Analysis ────────────────────────────────────────────────────────
    print("Running graph-search clustering …")
    results = grid_graph_search_clustering(
        qc=qc, xdiv=xdiv, ydiv=ydiv,
        session=session, use_all_sessions=use_all_sessions,
        use_normalized=use_normalized,
    )
    T = results["T"]

    q_label = (f"q = {qc:.2f}  (normalised r)"
               if use_normalized else f"q = {qc} m/s  (raw speed)")

    if key_timesteps is None:
        key_timesteps = [max(0, T // 6), T // 2, min(T - 1, 5 * T // 6)]
    key_timesteps = [min(max(t, 0), T - 1) for t in key_timesteps]

    # ── 2. Pre-compute simplified network & median speeds ──────────────────
    print("Building simplified network …")
    rep_links = _simplified_rep_links(results["bounds"], xdiv, ydiv)

    print("Computing median speeds per cluster …")
    median_spd = _compute_median_speeds(results, top_n=5)   # (T, 5)

    # ── 3. Standalone plots ────────────────────────────────────────────────
    plot_top5_cluster_sizes(results, folder)
    plot_median_speeds(results, folder)

    # ── 4. All-timestep grid map ───────────────────────────────────────────
    print("Generating all-timestep map grid …")
    #plot_all_timestep_maps(results, folder, rep_links)

    # ── 5. Individual maps ────────────────────────────────────────────────
    for t in key_timesteps:
        fig, ax = plt.subplots(dpi=200)
        _draw_cluster_map(ax, t, results, rep_links)
        ax.legend(handles=_cluster_legend_handles(), fontsize=4.5, loc="best", ncol=2)
        out = os.path.join(folder, f"graph_search_t{t}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved map t={t} → {out}")

    # ── 6. Merged figure  (3 rows) ────────────────────────────────────────
    n_maps    = len(key_timesteps)
    ts_labels = [_TS_BASE[i].strftime("%H:%M") for i in range(T)]
    tick_step = max(1, T // 10)

    fig = plt.figure(figsize=(4.5 * n_maps, 13), dpi=200)
    gs_layout = fig.add_gridspec(
        3, n_maps,
        height_ratios=[1.2, 0.6, 0.6],
        hspace=0.48,
        wspace=0.10,
    )

    # Row 0: maps
    for col, t in enumerate(key_timesteps):
        ax = fig.add_subplot(gs_layout[0, col])
        _draw_cluster_map(ax, t, results, rep_links)
        if col == 0:
            ax.legend(handles=_cluster_legend_handles(), fontsize=4,
                      loc="lower left", ncol=1)

    # Row 1: top-5 cluster sizes (per-timestep size rank)
    ax_sz = fig.add_subplot(gs_layout[1, :])
    top5  = _compute_ranked_sizes(results, top_n=5)   # (T, 5)

    for k in range(5):
        if top5[:, k].max() == 0:
            continue
        ax_sz.plot(range(T), top5[:, k],
                   color=_TOP5_COLORS[k],
                   linewidth=1.5 if k == 0 else 1.0,
                   marker="o", markersize=2,
                   label=f"Cluster {k + 1}")

    for t in key_timesteps:
        ax_sz.axvline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    ax_sz.set_xticks(range(0, T, tick_step))
    ax_sz.set_xticklabels(ts_labels[::tick_step], rotation=30, ha="right", fontsize=7)
    ax_sz.set_ylabel("# grid cells", fontsize=8)
    ax_sz.set_title("Top-5 Congested Cluster Sizes", fontsize=9)
    ax_sz.legend(fontsize=7, loc="upper left")
    ax_sz.grid(True, alpha=0.3)
    ax_sz.tick_params(axis="both", labelsize=7)

    # Row 2: median raw speed per cluster
    ax_spd = fig.add_subplot(gs_layout[2, :])

    for k in range(5):
        col_data = median_spd[:, k]
        if np.all(np.isnan(col_data)):
            continue
        ax_spd.plot(range(T), col_data,
                    color=_TOP5_COLORS[k],
                    linewidth=1.5 if k == 0 else 1.0,
                    marker="o", markersize=2,
                    label=f"Cluster {k + 1}")

    for t in key_timesteps:
        ax_spd.axvline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    ax_spd.set_xticks(range(0, T, tick_step))
    ax_spd.set_xticklabels(ts_labels[::tick_step], rotation=30, ha="right", fontsize=7)
    ax_spd.set_xlabel("Time", fontsize=8)
    ax_spd.set_ylabel("Median speed  (m/s)", fontsize=8)
    ax_spd.set_title("Median Raw Speed of Top-5 Congested Clusters", fontsize=9)
    ax_spd.legend(fontsize=7, loc="upper left")
    ax_spd.grid(True, alpha=0.3)
    ax_spd.tick_params(axis="both", labelsize=7)

    fig.suptitle(
        f"Graph-Search Cluster Evolution  "
        f"({xdiv}×{ydiv} grid,  {q_label},  mean over all sessions)",
        fontsize=11, fontweight="bold",
    )

    out = os.path.join(folder, "merged_graph_search_analysis.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved merged figure → {out}")

    return results
