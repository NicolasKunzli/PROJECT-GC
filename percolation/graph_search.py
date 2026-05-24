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
• figure/graph_search/graph_search_t<t>.png   — map at one timestep
• figure/graph_search/top5_cluster_sizes.png  — time-series of top-5 sizes
• figure/graph_search/merged_graph_search_analysis.png — combined figure
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque

from config import DL, links, LOCAL_FIGURE, NODATA_COLOR
from network.draw import sublink, polyg
from network.simplify import group_segments
from percolation.core import compute_normalized_speed
from processing.speed import fill_speed_nans, mean_over_sessions


# ── Constants ──────────────────────────────────────────────────────────────────

# Colours for top-5 ranked congested clusters (rank 0 = largest)
_TOP5_COLORS = ["#e6194b", "#4363d8", "#f58231", "#3cb44b", "#911eb4"]
_SMALL_CONG_COLOR = "#ffbbbb"   # faded pink for smaller congested clusters
_FUNC_COLOR       = "#33aa33"   # green for functional segments

# 8-neighbours: cardinal directions + all 4 diagonals.
# At the 20×16 grid resolution (~200–300 m per cell) diagonal cells share
# real road infrastructure, so 8-connectivity gives a more physically
# meaningful definition of a "contiguous congestion zone".
_NEIGHBOURS_8 = (
    (-1,  0), (1,  0), (0, -1), (0,  1),   # cardinal
    (-1, -1), (-1, 1), (1, -1), (1,  1),   # diagonal
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


def _assign_cell_state(r_t, nodata_mask, links_cells, xdiv, ydiv, qc):
    """
    Compute mean normalised-speed per cell and derive congestion state.

    Returns
    -------
    grid_state : ndarray (xdiv, ydiv), values  -1 = no-data, 0 = functional, 1 = congested
    """
    valid = (~nodata_mask) & (~np.isnan(r_t))

    df = pd.DataFrame({
        "cell_x": links_cells["cell_x"].values,
        "cell_y": links_cells["cell_y"].values,
        "r":      r_t,
        "valid":  valid,
    })

    cell_r = df[df["valid"]].groupby(["cell_x", "cell_y"])["r"].mean()

    grid_state = np.full((xdiv, ydiv), -1, dtype=int)
    for (cx, cy), rv in cell_r.items():
        cx, cy = int(cx), int(cy)
        if 0 <= cx < xdiv and 0 <= cy < ydiv:
            grid_state[cx, cy] = 1 if rv < qc else 0

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
                    for dr, dc in _NEIGHBOURS_8:   # ← 8 directions incl. diagonals
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


def _simplified_rep_links(bounds, xdiv, ydiv, distance_threshold=50.0):
    """
    Build representative-link DataFrame for the simplified network,
    annotated with grid-cell assignments.  Call once and reuse.
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
):
    """
    Run graph-search clustering on the grid for every timestep.

    Parameters
    ----------
    qc              : float – normalised-speed threshold (r̄_cell < qc → congested)
    xdiv, ydiv      : int   – grid divisions (columns, rows)
    session         : int   – session index used when use_all_sessions=False
    use_all_sessions: bool  – if True, average r over all sessions before analysis

    Returns
    -------
    dict with keys:
        'cong_sizes'  : list[list[int]]     – sorted cluster sizes (congested) per timestep
        'func_sizes'  : list[list[int]]     – sorted cluster sizes (functional) per timestep
        'grid_states' : list[ndarray]       – (xdiv, ydiv) state grids
        'cong_labels' : list[ndarray]       – scipy-label arrays for congested components
        'func_labels' : list[ndarray]       – scipy-label arrays for functional components
        'T'           : int
        'qc'          : float
        'xdiv','ydiv' : int
        'bounds'      : (x_min, x_max, y_min, y_max)
        'nodata_mask' : bool ndarray (N,)
    """
    r = compute_normalized_speed()                      # (S, T, N)

    if use_all_sessions:
        r_mean    = mean_over_sessions(r, min=0, max=r.shape[0])   # (T, N)
        r_profile, nodata_mask = fill_speed_nans(r_mean)
    else:
        r_profile, nodata_mask = fill_speed_nans(r[session])       # (T, N)

    T      = r_profile.shape[0]
    bounds = _network_bounds()
    lc     = _links_with_cells(bounds, xdiv, ydiv)

    cong_sizes_all, func_sizes_all = [], []
    grid_states, cong_labels_all, func_labels_all = [], [], []

    print(f"  Graph search: processing {T} timesteps …")
    for t in range(T):
        gs             = _assign_cell_state(r_profile[t], nodata_mask, lc, xdiv, ydiv, qc)
        cl, nc, fl, nf = _find_components(gs)

        cong_sizes_all.append(_sorted_sizes(cl, nc))
        func_sizes_all.append(_sorted_sizes(fl, nf))
        grid_states.append(gs)
        cong_labels_all.append(cl)
        func_labels_all.append(fl)

    print(f"  Done.  (qc={qc}, {xdiv}×{ydiv} grid, {T} timesteps)")

    return dict(
        cong_sizes=cong_sizes_all,
        func_sizes=func_sizes_all,
        grid_states=grid_states,
        cong_labels=cong_labels_all,
        func_labels=func_labels_all,
        T=T,
        qc=qc,
        xdiv=xdiv,
        ydiv=ydiv,
        bounds=bounds,
        nodata_mask=nodata_mask,
    )


# ── Time-series plot ───────────────────────────────────────────────────────────

def plot_top5_cluster_sizes(results, folder, fname="top5_cluster_sizes.png"):
    """
    Line chart: top-5 congested cluster sizes over time.

    Returns the output path.
    """
    T      = results["T"]
    qc     = results["qc"]
    xdiv   = results["xdiv"]
    ydiv   = results["ydiv"]
    sizes  = results["cong_sizes"]

    # Build (T, 5) matrix — pad with 0 when fewer than 5 clusters exist
    top5 = np.zeros((T, 5), dtype=int)
    for t, s in enumerate(sizes):
        for k, v in enumerate(s[:5]):
            top5[t, k] = v

    ts_labels = [_TS_BASE[i].strftime("%H:%M") for i in range(T)]
    tick_step = max(1, T // 10)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

    for k in range(5):
        if top5[:, k].max() == 0:
            continue
        ax.plot(
            range(T),
            top5[:, k],
            color=_TOP5_COLORS[k],
            linewidth=1.8 if k == 0 else 1.2,
            marker="o",
            markersize=3,
            label=f"Rank {k + 1}",
        )

    ax.set_xticks(range(0, T, tick_step))
    ax.set_xticklabels(ts_labels[::tick_step], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel("Cluster size  (# grid cells)", fontsize=9)
    ax.set_title(
        f"Top-5 Congested Cluster Sizes Over Time  "
        f"(q = {qc:.2f},  {xdiv}×{ydiv} grid,  mean over all sessions)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, fname)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved top-5 cluster sizes → {out}")
    return out


# ── Map drawing ────────────────────────────────────────────────────────────────

def _draw_cluster_map(ax, t, results, rep_links):
    """
    Draw the graph-search cluster map for timestep `t` on `ax`.

    Congested clusters are coloured by size-rank: top-5 get distinct colours
    (_TOP5_COLORS); smaller clusters get a faded pink.
    Functional cells are drawn in green.
    No-data cells/segments are drawn in NODATA_COLOR.

    Parameters
    ----------
    ax        : matplotlib Axes
    t         : int – timestep index
    results   : dict returned by grid_graph_search_clustering
    rep_links : DataFrame – simplified representative links with 'cell_x', 'cell_y'
    """
    xdiv  = results["xdiv"]
    ydiv  = results["ydiv"]
    bounds = results["bounds"]
    x_min, x_max, y_min, y_max = bounds
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    gs = results["grid_states"][t]
    cl = results["cong_labels"][t]

    # Rank congested components by size (largest → rank 0)
    nc = int(cl.max())
    if nc > 0:
        comp_sizes = [(i, int((cl == i).sum())) for i in range(1, nc + 1)]
        cong_rank  = {
            comp: rank
            for rank, (comp, _) in enumerate(sorted(comp_sizes, key=lambda x: -x[1]))
        }
    else:
        cong_rank = {}

    def _cong_color(comp_id):
        rank = cong_rank.get(comp_id, 99)
        return _TOP5_COLORS[rank] if rank < 5 else _SMALL_CONG_COLOR

    # ── Draw grid cell rectangles (faded background) ───────────────────────
    for cx in range(xdiv):
        for cy in range(ydiv):
            state = gs[cx, cy]
            x0, y0 = x_min + cx * w, y_min + cy * h
            if state == 1:
                fc    = _cong_color(cl[cx, cy])
                alpha = 0.28
            elif state == 0:
                fc    = "#b8ffb8"
                alpha = 0.18
            else:
                fc, alpha = "none", 0.0
            ax.add_patch(patches.Rectangle(
                (x0, y0), w, h,
                edgecolor="black", facecolor=fc, alpha=alpha,
                linewidth=0.3, zorder=1,
            ))

    # ── Draw simplified road segments coloured by cluster ─────────────────
    for _, row in rep_links.iterrows():
        cx = int(np.clip(row["cell_x"], 0, xdiv - 1))
        cy = int(np.clip(row["cell_y"], 0, ydiv - 1))
        state = gs[cx, cy]

        if state == -1:
            color, lw = NODATA_COLOR, 0.4
        elif state == 1:
            color, lw = _cong_color(cl[cx, cy]), 0.9
        else:
            color, lw = _FUNC_COLOR, 0.7

        xs, ys = sublink(row)
        ax.plot(xs, ys, c=color, linewidth=lw, zorder=2)

    # ── Axis labels ────────────────────────────────────────────────────────
    ts_str = _TS_BASE[t].strftime("%H:%M")
    nc_c   = int((gs == 1).sum())
    nc_f   = int((gs == 0).sum())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(
        f"t={t}  ({ts_str})\n"
        f"{nc_c} congested / {nc_f} functional cells",
        fontsize=7,
    )
    ax.tick_params(axis="both", labelsize=5)


def _cluster_legend_handles():
    """Return legend handles for the cluster map."""
    handles = [
        plt.Line2D([0], [0], color=_TOP5_COLORS[k], lw=2,
                   label=f"Congested cluster rank {k + 1}")
        for k in range(5)
    ]
    handles += [
        plt.Line2D([0], [0], color=_SMALL_CONG_COLOR, lw=2, label="Congested (small)"),
        plt.Line2D([0], [0], color=_FUNC_COLOR,        lw=2, label="Functional"),
        plt.Line2D([0], [0], color=NODATA_COLOR,        lw=2, label="No data"),
    ]
    return handles


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_graph_search_analysis(
    qc=0.52,
    xdiv=20,
    ydiv=16,
    key_timesteps=None,
    session=0,
    use_all_sessions=True,
    folder=None,
):
    """
    Full graph-search clustering pipeline:

    1. Compute grid states + connected components for all timesteps.
    2. Save top-5 congested cluster sizes time-series.
    3. Save individual maps at `key_timesteps`.
    4. Save a merged figure (maps in top row, time-series in bottom row).

    Parameters
    ----------
    qc, xdiv, ydiv      : see grid_graph_search_clustering
    key_timesteps       : list[int] – timesteps shown on maps (default: early/mid/late)
    session             : int       – used when use_all_sessions=False
    use_all_sessions    : bool
    folder              : str       – output directory (default: figure/graph_search)

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
    )
    T = results["T"]

    # Default key timesteps: early, mid, late thirds of the recording
    if key_timesteps is None:
        key_timesteps = [max(0, T // 6), T // 2, min(T - 1, 5 * T // 6)]
    key_timesteps = [min(max(t, 0), T - 1) for t in key_timesteps]

    # ── 2. Pre-compute simplified network (done once) ──────────────────────
    print("Building simplified network …")
    rep_links = _simplified_rep_links(results["bounds"], xdiv, ydiv)

    # ── 3. Top-5 cluster sizes time-series ────────────────────────────────
    plot_top5_cluster_sizes(results, folder)

    # ── 4. Individual maps ────────────────────────────────────────────────
    for t in key_timesteps:
        fig, ax = plt.subplots(dpi=200)
        _draw_cluster_map(ax, t, results, rep_links)
        ax.legend(handles=_cluster_legend_handles(), fontsize=4.5, loc="best", ncol=2)
        out = os.path.join(folder, f"graph_search_t{t}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved map t={t} → {out}")

    # ── 5. Merged figure ──────────────────────────────────────────────────
    n_maps = len(key_timesteps)
    fig = plt.figure(figsize=(4.5 * n_maps, 8.5), dpi=200)
    gs_layout = fig.add_gridspec(
        2, n_maps,
        height_ratios=[1.1, 0.7],
        hspace=0.38,
        wspace=0.10,
    )

    # Top row: maps
    for col, t in enumerate(key_timesteps):
        ax = fig.add_subplot(gs_layout[0, col])
        _draw_cluster_map(ax, t, results, rep_links)
        if col == 0:
            ax.legend(handles=_cluster_legend_handles(), fontsize=4, loc="lower left", ncol=1)

    # Bottom row: spanning time-series
    ax_ts = fig.add_subplot(gs_layout[1, :])
    top5  = np.zeros((T, 5), dtype=int)
    for t, s in enumerate(results["cong_sizes"]):
        for k, v in enumerate(s[:5]):
            top5[t, k] = v

    ts_labels = [_TS_BASE[i].strftime("%H:%M") for i in range(T)]
    tick_step = max(1, T // 10)

    for k in range(5):
        if top5[:, k].max() == 0:
            continue
        ax_ts.plot(
            range(T),
            top5[:, k],
            color=_TOP5_COLORS[k],
            linewidth=1.5 if k == 0 else 1.0,
            marker="o",
            markersize=2,
            label=f"Rank {k + 1}",
        )

    # Mark key timesteps with vertical dashed lines
    for t in key_timesteps:
        ax_ts.axvline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    ax_ts.set_xticks(range(0, T, tick_step))
    ax_ts.set_xticklabels(ts_labels[::tick_step], rotation=30, ha="right", fontsize=7)
    ax_ts.set_xlabel("Time", fontsize=8)
    ax_ts.set_ylabel("# grid cells", fontsize=8)
    ax_ts.set_title(
        f"Top-5 Congested Cluster Sizes  (q = {qc:.2f})",
        fontsize=9,
    )
    ax_ts.legend(fontsize=7, loc="upper left")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.tick_params(axis="both", labelsize=7)

    fig.suptitle(
        f"Graph-Search Cluster Evolution  "
        f"({xdiv}×{ydiv} grid,  q = {qc:.2f},  mean over all sessions)",
        fontsize=11,
        fontweight="bold",
    )

    out = os.path.join(folder, "merged_graph_search_analysis.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved merged figure → {out}")

    return results
