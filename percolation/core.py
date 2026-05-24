"""
percolation/core.py — Percolation analysis of the road network (Li et al. 2015).

The key idea: at threshold q, keep only "functional" segments (normalised speed r ≥ q)
and track the size of strongly connected components. The transition point q_c where the
second-largest component is maximal marks the onset of network-wide congestion.
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
        vdist, vtime,
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

    # ── Plot 2: network at q_c with bottlenecks highlighted ────────────────────
    # Grey segments fall into two distinct categories:
    #   • No-data  (nodata_mask[i] = True): >30 % NaN fraction → insufficient observations.
    #     These segments are excluded from the percolation analysis entirely.
    #   • Congested (r_t[i] < qc): the segment has data but its normalised speed falls
    #     below q_c, so it does not contribute to the giant functional component.
    # Both are non-functional, but they have different meanings.
    fig, ax = plt.subplots(dpi=250)

    # Exclude no-data segments from the functional set even if they happen to have
    # a valid r_t value at this timestep (they'd otherwise leak into cluster colours).
    functional = (r_t >= qc) & (~nodata_mask) & (~np.isnan(r_t))

    # Pass 1: draw every segment with its explicit state
    for i, row in links.iterrows():
        x, y = sublink(row)
        if nodata_mask[i] or np.isnan(r_t[i]):
            # True no-data: insufficient observations → gray
            ax.plot(x, y, c=NODATA_COLOR, linewidth=0.3, zorder=1)
        elif not functional[i]:
            # Congested: has data but r < q_c → dark red to distinguish from no-data
            ax.plot(x, y, c="#c0392b", linewidth=0.4, zorder=1)

    # Pass 2: paint functional segments by their cluster membership (top-5)
    func_idx = np.where(functional)[0]
    if len(func_idx) > 0:
        sub_adj        = adj_directed[np.ix_(func_idx, func_idx)]
        _, comp_labels = connected_components(sub_adj, directed=True, connection="strong")
        comp_sizes     = np.bincount(comp_labels)
        size_order     = np.argsort(comp_sizes)[::-1]
        palette        = ["green", "blue", "orange", "purple", "cyan"]

        for k, cluster_id in enumerate(size_order[:5]):
            members = func_idx[comp_labels == cluster_id]
            c = palette[k] if k < len(palette) else "#aaaaaa"
            for idx in members:
                x, y = sublink(links.iloc[idx])
                ax.plot(x, y, c=c, linewidth=0.5, zorder=2)

        # Remaining functional segments (ranks 6+) in light teal
        shown = set()
        for cluster_id in size_order[:5]:
            shown.update(func_idx[comp_labels == cluster_id].tolist())
        for idx in func_idx:
            if idx not in shown:
                x, y = sublink(links.iloc[idx])
                ax.plot(x, y, c="#7ecaca", linewidth=0.4, zorder=2)

    # Pass 3: highlight bottleneck segments
    for idx in bottlenecks:
        x, y = sublink(links.iloc[idx])
        ax.plot(x, y, c="red", linewidth=1.5, zorder=3)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color=p, lw=2, label=f"Functional cluster {k+1}")
        for k, p in enumerate(palette)
    ] + [
        plt.Line2D([0], [0], color="#c0392b",  lw=2, label=f"Congested  (r < {qc:.2f})"),
        plt.Line2D([0], [0], color=NODATA_COLOR, lw=2,
                   label="No data  (>30 % missing obs.)"),
        plt.Line2D([0], [0], color="red", lw=2, label=f"Bottleneck ({len(bottlenecks)})"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=2)

    ax.set_aspect("equal")
    ax.set_title(
        f"Percolation network at $q_c$={qc:.3f}\n"
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
