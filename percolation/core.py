"""
percolation/core.py — Percolation analysis of the road network (Li et al. 2015).

The key idea: at threshold q, keep only "functional" segments (normalised speed r ≥ q)
and track the size of strongly connected components. The transition point q_c where the
second-largest component is maximal marks the onset of network-wide congestion.

Normalisation:
    r_ij(t) = v_ij(t) / v_max_ij

    v_ij(t) is computed from vehicle distance over vehicle time, and v_max_ij is the
    95th-percentile speed of segment j across all sessions and timesteps.  This keeps
    the threshold less strict than using the absolute free-flow maximum.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from config import DL, links, connections, LOCAL_FIGURE, NODATA_COLOR
from network.draw import sublink
from processing.speed import fill_speed_nans, mean_over_sessions


def build_directed_adjacency():
    """Build a directed adjacency matrix (CSR) from the connections DataFrame."""
    N = len(links)
    row_idx, col_idx = [], []
    for _, row in connections.iterrows():
        org = DL.section_id_to_index.get(row["org"])
        dst = DL.section_id_to_index.get(row["dst"])
        if org is not None and dst is not None:
            row_idx.append(org)
            col_idx.append(dst)
    return csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(N, N))


def compute_normalized_speed():
    """
    Compute normalised speed r_ij(t) = v_ij(t) / v_max_ij for each segment.

    v_max_ij is the 95th-percentile speed of segment j across all sessions and timesteps,
    making r robust to short-lived noise. r is clipped to [0, 1].

    Returns ndarray of shape (S, T, N).
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist,
        vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=vtime != 0,
    )

    v_max = np.nanpercentile(speed.reshape(-1, speed.shape[2]), 95, axis=0)
    v_max[v_max == 0] = np.nan  # avoid division by zero

    r = np.clip(speed / v_max[np.newaxis, np.newaxis, :], 0, 1)
    return r


def percolation_sweep(r_t, adj_directed, q_values):
    """
    Sweep over q values for a single time snapshot r_t (shape N,).

    At each q, keep only segments with r_t ≥ q (functional subgraph) and
    find strongly connected components.

    Returns
    -------
    giant  : ndarray – giant component size at each q
    second : ndarray – second-largest component size at each q
    """
    giant  = np.zeros(len(q_values))
    second = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        func_idx = np.where(r_t >= q)[0]
        if len(func_idx) == 0:
            continue

        sub_adj     = adj_directed[np.ix_(func_idx, func_idx)]
        _, labels   = connected_components(sub_adj, directed=True, connection="strong")
        comp_sizes  = np.sort(np.bincount(labels))[::-1]
        giant[i]    = comp_sizes[0]
        if len(comp_sizes) > 1:
            second[i] = comp_sizes[1]

    return giant, second


def find_critical_threshold(r_t, adj_directed, q_values):
    """
    Find q_c: the threshold where the second-largest component is maximal.

    Returns (q_c, giant_sizes, second_sizes).
    """
    giant, second = percolation_sweep(r_t, adj_directed, q_values)
    qc_idx = np.argmax(second)
    return q_values[qc_idx], giant, second


def find_bottlenecks(r_t, adj_directed, qc, delta=0.01):
    """
    Identify bottleneck segments: functional just below q_c but dysfunctional just above.

    Bottlenecks are the segments whose removal causes the network to fragment.

    Returns int ndarray of bottleneck segment indices.
    """
    bottlenecks = np.where((r_t >= qc - delta) & ~(r_t >= qc + delta))[0]
    return bottlenecks


def find_top_clusters(r_t, adj_directed, qc, n=5):
    """
    Return segment indices for the top-n strongly connected components at qc.

    Returns a list of arrays (one per cluster), sorted by descending component size.
    """
    func_idx = np.where(r_t >= qc)[0]
    if len(func_idx) == 0:
        return []

    sub_adj = adj_directed[np.ix_(func_idx, func_idx)]
    _, comp_labels = connected_components(sub_adj, directed=True, connection="strong")
    comp_sizes = np.bincount(comp_labels)
    size_order = np.argsort(comp_sizes)[::-1]

    return [
        func_idx[comp_labels == size_order[k]]
        for k in range(min(n, len(size_order)))
    ]


def percolation_analysis(session=0, timestep=None, n_q=100):
    """
    Full percolation analysis pipeline.

    Computes q_c, plots the percolation transition curve, and produces a network
    map with bottleneck segments highlighted in red.

    Parameters
    ----------
    session  : int – simulation session index
    timestep : int or None – if None, averages r over all sessions and timesteps
    n_q      : int – number of q values to sweep

    Returns
    -------
    (q_c, bottlenecks) : (float, int ndarray)
    """
    r            = compute_normalized_speed()
    adj_directed = build_directed_adjacency()
    q_values     = np.linspace(0, 1, n_q)

    if timestep is not None:
        r_session_profile, nodata_mask = fill_speed_nans(r[session])
        r_t        = r_session_profile[timestep]
        time_label = f"session {session}, t={timestep}"
    else:
        r_profile, nodata_mask = fill_speed_nans(
            mean_over_sessions(r, min=0, max=r.shape[0])
        )
        r_t        = np.nanmean(r_profile, axis=0)
        time_label = "mean over all sessions/timesteps"

    qc, giant, second = find_critical_threshold(r_t, adj_directed, q_values)
    print(f"Critical threshold q_c = {qc:.3f} ({time_label})")

    n_nodata = int(nodata_mask.sum())
    if n_nodata:
        print(f"  ({n_nodata} segments flagged no-data and rendered in gray)")

    bottlenecks = find_bottlenecks(r_t, adj_directed, qc)
    print(f"Found {len(bottlenecks)} bottleneck segments at q_c")

    # ── Plot 1: giant & second-largest component vs q ──────────────────────────
    fig, ax = plt.subplots(dpi=250)
    ax.plot(q_values, giant,  label="Giant component (G)",  color="blue")
    ax.plot(q_values, second, label="2nd largest (SG)",     color="red")
    ax.axvline(qc, color="gray", linestyle="--", linewidth=0.8, label=f"$q_c$ = {qc:.3f}")
    ax.set_xlabel("Threshold q", fontsize=10)
    ax.set_ylabel("Component size (# segments)", fontsize=10)
    ax.set_title("Percolation transition", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    fig.savefig(f"{LOCAL_FIGURE}/percolation_transition.png")
    plt.close(fig)
    print(f"Saved {LOCAL_FIGURE}/percolation_transition.png")

    # ── Plot 2: simplified network at q_c with bottlenecks highlighted ────────
    # The percolation math (SCC, q_c, bottlenecks) was computed on the full graph.
    # Visualisation uses the *simplified* graph: parallel/duplicate road segments
    # are merged into one representative (highest lane count), coloured by their
    # group-mean r.  This removes visual clutter from dual-carriageways while
    # preserving the spatial structure of the percolation clusters.
    from network.simplify import group_segments

    palette = ["green", "blue", "orange", "purple", "cyan"]

    # ── Build segment → SCC-rank lookup (from full-network percolation) ────
    functional   = (r_t >= qc) & (~nodata_mask) & (~np.isnan(r_t))
    func_idx     = np.where(functional)[0]
    seg_to_rank  = {}  # segment index → cluster rank (0 = giant, 1 = 2nd, …)

    if len(func_idx) > 0:
        sub_adj      = adj_directed[np.ix_(func_idx, func_idx)]
        _, comp_labels = connected_components(sub_adj, directed=True, connection="strong")
        comp_sizes   = np.bincount(comp_labels)
        size_order   = np.argsort(comp_sizes)[::-1]
        label_to_rank = {label: rank for rank, label in enumerate(size_order)}
        for local_i, seg_idx in enumerate(func_idx):
            seg_to_rank[int(seg_idx)] = label_to_rank[comp_labels[local_i]]

    # ── Build simplified groups ────────────────────────────────────────────
    groups = group_segments(distance_threshold=50.0)
    reps   = [max(g, key=lambda idx: links.iloc[idx]["num_lanes"]) for g in groups]

    # Group-level r = nanmean of member r values (nodata members excluded)
    bottleneck_set = set(int(b) for b in bottlenecks)
    group_r            = []
    group_nodata       = []
    group_is_bottleneck = []
    group_rank         = []

    for group in groups:
        valid_r = [r_t[i] for i in group if not nodata_mask[i] and not np.isnan(r_t[i])]
        r_grp   = float(np.mean(valid_r)) if valid_r else np.nan
        group_r.append(r_grp)
        group_nodata.append(np.isnan(r_grp))
        group_is_bottleneck.append(any(i in bottleneck_set for i in group))
        # Cluster rank: use the representative segment's rank (fallback: 5 = teal)
        group_rank.append(seg_to_rank.get(reps[len(group_r) - 1], 5))

    n_bottleneck_groups = sum(group_is_bottleneck)

    fig, ax = plt.subplots(dpi=250)

    # Pass 1 & 2: draw every representative segment
    for k, rep_idx in enumerate(reps):
        x, y = sublink(links.iloc[rep_idx])
        if group_nodata[k]:
            ax.plot(x, y, c=NODATA_COLOR, linewidth=0.3, zorder=1)
        elif group_r[k] < qc:
            # Congested barrier
            ax.plot(x, y, c="#c0392b", linewidth=1.0, zorder=2)
        else:
            # Functional — colour by SCC rank
            rank = group_rank[k]
            c    = palette[rank] if rank < len(palette) else "#7ecaca"
            ax.plot(x, y, c=c, linewidth=0.6, zorder=3)

    # Pass 3: overlay bottleneck representatives in red
    for k, rep_idx in enumerate(reps):
        if group_is_bottleneck[k]:
            x, y = sublink(links.iloc[rep_idx])
            ax.plot(x, y, c="red", linewidth=1.8, zorder=4)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color=p, lw=2, label=f"Functional cluster {k+1}")
        for k, p in enumerate(palette)
    ] + [
        plt.Line2D([0], [0], color="#c0392b",  lw=2, label=f"Congested  (r < {qc:.2f})"),
        plt.Line2D([0], [0], color=NODATA_COLOR, lw=2,
                   label="No data  (>30 % missing obs.)"),
        plt.Line2D([0], [0], color="red", lw=2,
                   label=f"Bottleneck ({n_bottleneck_groups} groups)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=2)

    ax.set_aspect("equal")
    ax.set_title(
        f"Percolation network at $q_c$={qc:.3f}  (simplified graph)\n"
        f"Dark red = congested  |  Gray = no data  |  Colors = functional clusters  |  Red = bottleneck",
        fontsize=8,
    )
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    fig.savefig(f"{LOCAL_FIGURE}/percolation_bottlenecks.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LOCAL_FIGURE}/percolation_bottlenecks.png")

    return qc, bottlenecks
