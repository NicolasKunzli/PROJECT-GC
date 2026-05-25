"""
percolation/maps.py — Congestion visualisation built on percolation results.

congestion_map  : per-timestep maps showing red/green congestion status.
grid_clust      : rectangular grid overlay with per-cell congestion colour.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import cdist


from config import DL, links, LOCAL_FIGURE, NODATA_COLOR
from network.draw import sublink, polyg
from percolation.core import (
    build_directed_adjacency,
    compute_normalized_speed,
    find_critical_threshold,
)
from processing.speed import fill_speed_nans, mean_over_sessions


CONGESTED_COLOR = "red"
FUNCTIONAL_COLOR = "green"

_CLUSTER_PALETTE = ["#e6194b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _network_bounds(tol=100):
    x_min = min(np.min(links["from_x"]), np.min(links["to_x"])) - tol
    x_max = max(np.max(links["from_x"]), np.max(links["to_x"])) + tol
    y_min = min(np.min(links["from_y"]), np.min(links["to_y"])) - tol
    y_max = max(np.max(links["from_y"]), np.max(links["to_y"])) + tol
    return x_min, x_max, y_min, y_max


def _raw_speed_snapshot(session, timestep):
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist,
        vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=(vtime != 0) & (~np.isnan(vtime)),
    )

    speed_profile, nodata_mask = fill_speed_nans(speed[session])
    return speed_profile[timestep], nodata_mask


def _normalized_speed_snapshot(session, timestep):
    r = compute_normalized_speed()
    r_session_profile, nodata_mask = fill_speed_nans(r[session])
    return r_session_profile[timestep], nodata_mask


def _finish_map_axis(ax, title, bounds):
    x_min, x_max, y_min, y_max = bounds
    polyg(ax, color="black", alpha=0.3, zorder=-1)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", fontsize=8)
    ax.set_ylabel("Y [m]", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)


def _draw_segment_congestion(ax, values, nodata_mask, threshold, title, bounds):
    valid = ~(nodata_mask | np.isnan(values))
    congested = (values < threshold) & valid

    for i, row in links.iterrows():
        x, y = sublink(row)
        if not valid[i]:
            color = NODATA_COLOR
            zorder = 0
        elif congested[i]:
            color = CONGESTED_COLOR
            zorder = 2
        else:
            color = FUNCTIONAL_COLOR
            zorder = 1
        ax.plot(x, y, c=color, linewidth=1, zorder=zorder)

    _finish_map_axis(ax, f"{title}\n{int(congested.sum())}/{int(valid.sum())} congested", bounds)
    return congested


def _draw_grid_congestion(ax, values, nodata_mask, threshold, xdiv, ydiv, title, bounds):
    x_min, x_max, y_min, y_max = bounds
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    links_local = links.copy()
    links_local["cell_x"] = np.clip(((links_local["c_x"] - x_min) // w).astype(int), 0, xdiv - 1)
    links_local["cell_y"] = np.clip(((links_local["c_y"] - y_min) // h).astype(int), 0, ydiv - 1)
    links_local["value"] = values
    links_local["nodata"] = nodata_mask | np.isnan(values)

    cell_values = (
        links_local.loc[~links_local["nodata"]]
        .groupby(["cell_x", "cell_y"])["value"]
        .mean()
        .reset_index(name="cell_value")
    )
    links_local = links_local.merge(cell_values, on=["cell_x", "cell_y"], how="left")

    valid = (~links_local["nodata"] & ~links_local["cell_value"].isna()).to_numpy()
    congested = (links_local["cell_value"].to_numpy() < threshold) & valid

    for _, row in links_local.iterrows():
        if row["nodata"] or np.isnan(row["cell_value"]):
            color = NODATA_COLOR
            zorder = 0
        elif row["cell_value"] < threshold:
            color = CONGESTED_COLOR
            zorder = 2
        else:
            color = FUNCTIONAL_COLOR
            zorder = 1
        ax.plot([row["from_x"], row["to_x"]], [row["from_y"], row["to_y"]],
                color=color, linewidth=1, zorder=zorder)

    for ix in range(xdiv):
        for iy in range(ydiv):
            ax.add_patch(patches.Rectangle(
                (x_min + ix * w, y_min + iy * h),
                w,
                h,
                edgecolor="black",
                facecolor="none",
                linewidth=0.4,
                alpha=0.5,
                zorder=3,
            ))

    _finish_map_axis(ax, f"{title}\n{int(congested.sum())}/{int(valid.sum())} links in congested cells", bounds)
    return congested


def _build_simplified_r_at_t(session, timestep):
    """
    Return (rep_df, qc_auto) for the simplified network at a given snapshot.

    rep_df columns: all link columns + 'r' (group mean normalised speed),
                    'nodata' (bool), 'cell_x', 'cell_y' (set externally).

    The normalised speed r for each group is the mean of its members' r values
    (excluding members flagged as no-data or NaN).
    """
    from network.simplify import group_segments

    norm_speed, norm_nodata = _normalized_speed_snapshot(session, timestep)

    groups = group_segments(distance_threshold=50.0)
    reps   = [max(g, key=lambda idx: links.iloc[idx]["num_lanes"]) for g in groups]

    group_r = []
    for group in groups:
        valid_r = [
            norm_speed[i]
            for i in group
            if not norm_nodata[i] and not np.isnan(norm_speed[i])
        ]
        group_r.append(float(np.mean(valid_r)) if valid_r else np.nan)

    rep_df = links.iloc[reps].copy().reset_index(drop=True)
    rep_df["r"]      = group_r
    rep_df["nodata"] = np.isnan(rep_df["r"])

    return rep_df, norm_speed, norm_nodata


def compare_congestion_methods(
    session=0,
    timestep=31,
    xdiv=20,
    ydiv=16,
    qc=None,
    grid_qc=0.52,
    n_q=100,
    output=None,
    # Legacy parameters kept for call-site compatibility (ignored)
    percentile=30,
    grid_percentile=50,
    speed_threshold=None,
    grid_threshold=None,
):
    """
    Side-by-side congestion maps using the **simplified road network** for both panels.

    Left  panel — Grid aggregation
        The normalised speed r is averaged per grid cell (xdiv × ydiv).
        Cells with mean r < grid_qc are congested.  Uses the simplified network.

    Right panel — Percolation
        Each simplified-network segment is coloured by its normalised speed vs q_c.
        q_c is computed from the full network's percolation sweep if not supplied.

    Parameters
    ----------
    session   : int        – simulation session index
    timestep  : int        – 3-minute timestep index
    xdiv,ydiv : int        – grid divisions for the aggregation panel
    qc        : float/None – percolation q_c; computed automatically if None
    grid_qc   : float      – normalised-speed threshold for grid aggregation (default 0.52)
    n_q       : int        – resolution of the q sweep when qc is computed
    output    : str/None   – optional output path override
    """
    bounds = _network_bounds()
    x_min, x_max, y_min, y_max = bounds
    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    # ── Compute qc if needed (full network, unfiltered) ────────────────────
    norm_speed_full, norm_nodata_full = _normalized_speed_snapshot(session, timestep)
    if qc is None:
        adj_directed = build_directed_adjacency()
        q_values     = np.linspace(0, 1, n_q)
        qc, _, _     = find_critical_threshold(norm_speed_full, adj_directed, q_values)

    # ── Build simplified network snapshot ─────────────────────────────────
    rep_df, _, _ = _build_simplified_r_at_t(session, timestep)

    # Grid-cell assignment for rep segments
    rep_df["cell_x"] = np.clip(
        ((rep_df["c_x"] - x_min) // w).astype(int), 0, xdiv - 1
    )
    rep_df["cell_y"] = np.clip(
        ((rep_df["c_y"] - y_min) // h).astype(int), 0, ydiv - 1
    )

    # Cell mean r (for grid panel)
    cell_r = (
        rep_df[~rep_df["nodata"]]
        .groupby(["cell_x", "cell_y"])["r"]
        .mean()
        .reset_index(name="cell_r")
    )
    rep_grid = rep_df.merge(cell_r, on=["cell_x", "cell_y"], how="left")

    # ── Output path ────────────────────────────────────────────────────────
    if output is None:
        folder = f"{LOCAL_FIGURE}/congestion_maps"
        os.makedirs(folder, exist_ok=True)
        output = f"{folder}/compare_methods_t{timestep}.png"
    else:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=250)

    # ── LEFT: Grid aggregation (simplified network, r vs grid_qc) ─────────
    ax = axes[0]
    n_cong_grid, n_valid_grid = 0, 0

    for _, row in rep_grid.iterrows():
        xs, ys = sublink(row)
        cell_rv = row.get("cell_r", np.nan)
        if row["nodata"] or np.isnan(cell_rv):
            color = NODATA_COLOR
        elif cell_rv < grid_qc:
            color = CONGESTED_COLOR
            n_cong_grid += 1
            n_valid_grid += 1
        else:
            color = FUNCTIONAL_COLOR
            n_valid_grid += 1
        ax.plot(xs, ys, c=color, linewidth=1, zorder=2)

    for ix in range(xdiv):
        for iy in range(ydiv):
            ax.add_patch(patches.Rectangle(
                (x_min + ix * w, y_min + iy * h), w, h,
                edgecolor="black", facecolor="none",
                linewidth=0.4, alpha=0.5, zorder=3,
            ))

    _finish_map_axis(
        ax,
        (
            f"Grid aggregation (simplified network)\n"
            f"{xdiv}×{ydiv}, cell mean r < {grid_qc:.2f} → congested\n"
            f"{n_cong_grid}/{n_valid_grid} links in congested cells"
        ),
        bounds,
    )

    # ── RIGHT: Percolation (simplified network, r vs qc) ──────────────────
    ax = axes[1]
    n_cong_perc, n_valid_perc = 0, 0

    for _, row in rep_df.iterrows():
        xs, ys = sublink(row)
        if row["nodata"]:
            color = NODATA_COLOR
        elif row["r"] < qc:
            color = CONGESTED_COLOR
            n_cong_perc += 1
            n_valid_perc += 1
        else:
            color = FUNCTIONAL_COLOR
            n_valid_perc += 1
        ax.plot(xs, ys, c=color, linewidth=1, zorder=2)

    _finish_map_axis(
        ax,
        (
            f"Percolation (simplified network)\n"
            f"r < q_c = {qc:.3f}\n"
            f"{n_cong_perc}/{n_valid_perc} links congested"
        ),
        bounds,
    )

    # ── Legend + title ─────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=CONGESTED_COLOR, lw=3, label="Congested"),
        plt.Line2D([0], [0], color=FUNCTIONAL_COLOR, lw=3, label="Functional"),
        plt.Line2D([0], [0], color=NODATA_COLOR,     lw=3, label="No data  (>30 % missing)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        f"Congestion comparison — session {session}, t={timestep} ({timestep * 3} min)",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.18, wspace=0.14)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved congestion-method comparison → {output}")
    return output


def congestion_map(qc, session=0, timesteps=None):
    """
    Plot congestion maps at selected timesteps using the percolation threshold q_c.

    Segments with normalised speed r < q_c are congested (red); r ≥ q_c are functional
    (green); no-data segments are rendered in NODATA_COLOR (gray).

    Parameters
    ----------
    qc        : float – critical threshold from percolation_analysis
    session   : int   – simulation session index
    timesteps : list  – timestep indices to plot; defaults to [0, 9, 23, 31, 33, 36, 38]
    """
    if timesteps is None:
        timesteps = [0, 9, 23, 31, 33, 36, 38]

    r      = compute_normalized_speed()
    folder = f"{LOCAL_FIGURE}/congestion_maps"
    os.makedirs(folder, exist_ok=True)

    r_session_profile, nodata_mask = fill_speed_nans(r[session])

    for t in timesteps:
        r_t = r_session_profile[t]

        fig, ax = plt.subplots(dpi=250)
        for i, row in links.iterrows():
            x, y = sublink(row)
            if nodata_mask[i] or np.isnan(r_t[i]):
                ax.plot(x, y, c=NODATA_COLOR, linewidth=1, zorder=0)
            elif r_t[i] < qc:
                ax.plot(x, y, c="red",   linewidth=1, zorder=2)
            else:
                ax.plot(x, y, c="green", linewidth=1, zorder=1)

        polyg(ax, color="black", alpha=0.3, zorder=-1)

        valid      = ~(nodata_mask | np.isnan(r_t))
        n_congested = int(np.sum((r_t < qc) & valid))
        ax.set_aspect("equal")
        ax.set_title(
            f"Congestion at t={t} ({t*3}min) — {n_congested}/{len(links)} congested, $q_c$={qc:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

        handles = [
            plt.Line2D([0], [0], color="red",   lw=2, label=f"Congested (r < {qc:.2f})"),
            plt.Line2D([0], [0], color="green", lw=2, label=f"Functional (r ≥ {qc:.2f})"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")

        fig.savefig(f"{folder}/congestion_t{t}.png")
        plt.close(fig)
        print(f"Saved congestion map t={t} ({t*3}min): {n_congested} congested segments")


def grid_clust(xdiv=4, ydiv=4, percentile=65, qc=None, session_min=0, session_max=100):
    """
    Cluster links by rectangular grid and colour each cell by its congestion level.

    Two modes:
      - qc is None   : raw-speed percentile mode — cells with mean speed below the
                        `percentile`-th percentile are red, above are green.
      - qc provided  : percolation mode — cells with mean normalised speed r < q_c
                        are red (congested), ≥ q_c are green (functional).

    Parameters
    ----------
    xdiv, ydiv  : int   – grid divisions along x and y
    percentile  : int   – raw-speed percentile threshold (ignored when qc is set)
    qc          : float or None – percolation critical threshold
    session_min, session_max : int – session slice (used in raw-speed mode)
    """
    links_local = links.copy()
    tol   = 100
    x_min = np.min(links_local["from_x"]) - tol
    x_max = np.max(links_local["to_x"])   + tol
    y_min = np.min(links_local["from_y"]) - tol
    y_max = np.max(links_local["to_y"])   + tol

    w  = (x_max - x_min) / xdiv
    h  = (y_max - y_min) / ydiv
    xs = np.arange(x_min, x_max, w)
    ys = np.arange(y_min, y_max, h)

    folder = "figure/clustering/grid_clusters"
    os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots(dpi=250)

    links_local["cell_x"] = ((links_local["c_x"] - x_min) // w).astype(int)
    links_local["cell_y"] = ((links_local["c_y"] - y_min) // h).astype(int)

    if qc is not None:
        # ── Percolation mode ───────────────────────────────────────────────────
        r         = compute_normalized_speed()
        r_profile, nodata_mask = fill_speed_nans(
            mean_over_sessions(r, min=0, max=r.shape[0])
        )
        r_mean = np.nanmean(r_profile, axis=0)

        links_local["speed_mean"] = r_mean
        links_local["nodata"]     = nodata_mask

        cell_speed  = (links_local.groupby(["cell_x", "cell_y"])["speed_mean"]
                       .mean().reset_index(name="cell_avg_speed"))
        links_local = links_local.merge(cell_speed, on=["cell_x", "cell_y"], how="left")

        links_local["color"] = np.where(links_local["cell_avg_speed"] >= qc, "green", "red")
        links_local.loc[links_local["nodata"], "color"] = NODATA_COLOR
        title = f"Percolation $q_c$ = {qc:.3f} (normalized speed)"
        fname = f"{folder}/grid_perc_qc{qc:.3f}.png"

    else:
        # ── Raw-speed percentile mode ──────────────────────────────────────────
        vdist = DL._vdist_3min.astype(float)
        vtime = DL._vtime_3min.astype(float)

        speed = np.divide(
            vdist, vtime,
            out=np.full(vdist.shape, np.nan),
            where=(vtime != 0) & (~np.isnan(vtime)),
        )
        speed_profile, nodata_mask = fill_speed_nans(
            mean_over_sessions(speed, min=session_min, max=session_max)
        )
        print(speed_profile.shape)

        avg_speed_per_link = np.nanmean(speed_profile, axis=0)
        links_local["speed_mean"] = avg_speed_per_link
        links_local["nodata"]     = nodata_mask

        cell_speed  = (links_local.groupby(["cell_x", "cell_y"])["speed_mean"]
                       .mean().reset_index(name="cell_avg_speed"))
        links_local = links_local.merge(cell_speed, on=["cell_x", "cell_y"], how="left")

        perc = np.nanpercentile(avg_speed_per_link, percentile)
        print(perc)

        links_local["color"] = np.where(links_local["cell_avg_speed"] >= perc, "green", "red")
        links_local.loc[links_local["nodata"], "color"] = NODATA_COLOR
        title = f"{percentile}th percentile : Speed = {perc:.2f} m/s"
        fname = f"{folder}/grid{percentile}.png"

    # ── Draw links ─────────────────────────────────────────────────────────────
    for _, row in links_local.iterrows():
        ax.plot([row["from_x"], row["to_x"]], [row["from_y"], row["to_y"]],
                color=row["color"])

    # ── Draw grid cells ────────────────────────────────────────────────────────
    for x in xs:
        for y in ys:
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                edgecolor="black", facecolor="none", linewidth=0.5,
            ))

    polyg(ax, color="black", zorder=-2)

    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    fig.savefig(fname)
    plt.close()


def grid_clust_kmeans(xdiv=20, ydiv=16, k=5, qc=None, percentile=65,
                      distance_threshold=50.0):
    """
    Grid K-means clustering on the simplified road network.

    Uses group_segments + compute_group_speeds so each drawn segment represents
    one physical road (duplicates merged), and speed is the group mean [m/s].

    Each cluster is labelled congested (mean speed < threshold) or functional.
    Congested cells are hatched; each cluster gets a distinct colour.

    Parameters
    ----------
    xdiv, ydiv           : int   – grid divisions
    k                    : int   – number of K-means clusters (≤ 5 recommended)
    qc                   : float or None – normalised speed threshold (percolation);
                           when set, group speeds are normalised by their 95th pct
    percentile           : int   – raw-speed percentile threshold (used when qc=None)
    distance_threshold   : float – segment-grouping radius passed to group_segments
    """
    from sklearn.cluster import KMeans
    import matplotlib.patches as mpatches
    from network.simplify import group_segments, compute_group_speeds

    # ── Simplified network ────────────────────────────────────────────────────
    groups          = group_segments(distance_threshold)
    representatives = [max(g, key=lambda idx: links.iloc[idx]["num_lanes"])
                       for g in groups]
    group_speeds    = compute_group_speeds(groups)   # m/s, shape (n_groups,)

    rep_link_indices = np.array(representatives)   # original indices into links
    rep_links = links.iloc[representatives].copy().reset_index(drop=True)
    rep_links["link_idx"]  = rep_link_indices
    rep_links["speed_mean"] = group_speeds
    rep_links["nodata"]     = np.isnan(group_speeds)

    # ── Grid bounds ───────────────────────────────────────────────────────────
    tol   = 100
    x_min = np.min(links["from_x"]) - tol
    x_max = np.max(links["to_x"])   + tol
    y_min = np.min(links["from_y"]) - tol
    y_max = np.max(links["to_y"])   + tol

    w  = (x_max - x_min) / xdiv
    h  = (y_max - y_min) / ydiv
    xs = np.arange(x_min, x_max, w)
    ys = np.arange(y_min, y_max, h)

    folder = "figure/clustering/grid_clusters"
    os.makedirs(folder, exist_ok=True)

    rep_links["cell_x"] = ((rep_links["c_x"] - x_min) // w).astype(int)
    rep_links["cell_y"] = ((rep_links["c_y"] - y_min) // h).astype(int)

    # ── Per-cell average speed + threshold ───────────────────────────────────
    if qc is not None:
        v_max = np.nanpercentile(group_speeds, 95)
        rep_links["speed_norm"] = np.clip(rep_links["speed_mean"] / v_max, 0, 1)
        cell_speed = (rep_links.groupby(["cell_x", "cell_y"])["speed_norm"]
                      .mean().reset_index(name="cell_avg_speed"))
        threshold  = qc
        title_mode = f"$q_c$={qc:.3f}"
        fname      = f"{folder}/grid_kmeans{k}_qc{qc:.3f}.png"
    else:
        cell_speed = (rep_links.groupby(["cell_x", "cell_y"])["speed_mean"]
                      .mean().reset_index(name="cell_avg_speed"))
        threshold  = np.nanpercentile(group_speeds, percentile)
        title_mode = f"{percentile}th pct"
        fname      = f"{folder}/grid_kmeans{k}_{percentile}pct.png"

    # ── K-means on normalised (cell_x, cell_y, speed) ─────────────────────────
    cells = cell_speed.dropna(subset=["cell_avg_speed"]).copy()

    spd_min   = cells["cell_avg_speed"].min()
    spd_range = cells["cell_avg_speed"].max() - spd_min or 1.0
    X = np.column_stack([
        cells["cell_x"] / max(xdiv - 1, 1),
        cells["cell_y"] / max(ydiv - 1, 1),
        (cells["cell_avg_speed"] - spd_min) / spd_range,
    ])

    n_clusters = min(k, len(cells))
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, init="k-means++")
    cells["cluster"] = km.fit_predict(X)

    dist = cdist(km.cluster_centers_, X)
    initial_cell_indices = np.argmin(dist, axis=1)

    cluster_mean_speed   = cells.groupby("cluster")["cell_avg_speed"].mean()
    cluster_is_congested = cluster_mean_speed < threshold

    cluster_color = {
        c: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        for i, c in enumerate(sorted(cells["cluster"].unique()))
    }

    # ── Merge cluster labels back to representative segments ──────────────────
    rep_links = rep_links.merge(
        cells[["cell_x", "cell_y", "cluster"]], on=["cell_x", "cell_y"], how="left"
    )
    rep_links["color"] = rep_links["cluster"].map(cluster_color).fillna(NODATA_COLOR)
    rep_links.loc[rep_links["nodata"], "color"] = NODATA_COLOR

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(dpi=250)

    for _, row in rep_links.iterrows():
        x, y = sublink(row)
        ax.plot(x, y, color=row["color"], linewidth=0.8)

    cell_lookup = cells.set_index(["cell_x", "cell_y"])
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            key = (xi, yi)
            if key in cell_lookup.index:
                cell_row  = cell_lookup.loc[key]
                clr       = cluster_color[cell_row["cluster"]]
                hatch     = "///" if cluster_is_congested[cell_row["cluster"]] else ""
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h,
                    edgecolor="black", facecolor=clr, alpha=0.2,
                    linewidth=0.5, hatch=hatch,
                ))
            else:
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h,
                    edgecolor="black", facecolor="none", linewidth=0.5,
                ))

    handles = [
        mpatches.Patch(
            facecolor=cluster_color[c], edgecolor="black",
            hatch="///" if cluster_is_congested[c] else "",
            label=f"Cluster {c + 1} — {'congested' if cluster_is_congested[c] else 'functional'}",
        )
        for c in sorted(cells["cluster"].unique())
    ]
    ax.legend(handles=handles, fontsize=6, loc="best")

    polyg(ax, color="black", zorder=-2)
    ax.set_title(f"Grid K-means k={k}, {title_mode}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    fig.savefig(fname, bbox_inches="tight")
    plt.close()

    # ── Median speed per cluster over time ────────────────────────────────────
    from clustering.kmeans import plot_cluster_speed
    import matplotlib.colors as mcolors

    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)
    speed = np.divide(vdist, vtime,
                      out=np.full(vdist.shape, np.nan), where=vtime != 0)

    T             = speed.shape[1]
    cluster_ids   = sorted(cells["cluster"].unique())
    timesteps_min = np.arange(T) * 3

    median_speed_time = np.full((T, len(cluster_ids)), np.nan)
    for ci, c in enumerate(cluster_ids):
        link_indices = rep_links.loc[rep_links["cluster"] == c, "link_idx"].values
        if len(link_indices) == 0:
            continue
        speed_c = speed[:, :, link_indices]
        for t in range(T):
            median_speed_time[t, ci] = np.nanmedian(speed_c[:, t, :])

    colors_rgba = [mcolors.to_rgba(cluster_color[c]) for c in cluster_ids]
    graph_name  = f"grid_kmeans{k}_{title_mode}_median"

    plot_cluster_speed(
        timesteps=timesteps_min,
        values=median_speed_time,
        name=graph_name,
        folder=folder,
        colors=colors_rgba,
        ylabel="Median speed (m/s)",
    )

    initial_cells = cells.iloc[initial_cell_indices][["cell_x", "cell_y"]]

    starting_links = []

    for _, cell in initial_cells.iterrows():
        links_in_cell = rep_links[
            (rep_links["cell_x"] == cell["cell_x"]) &
            (rep_links["cell_y"] == cell["cell_y"])
        ].copy()

        links_in_cell["dist_to_cell_center"] = np.sqrt(
            (links_in_cell["c_x"] - (x_min + (cell["cell_x"] + 0.5) * w)) ** 2 +
            (links_in_cell["c_y"] - (y_min + (cell["cell_y"] + 0.5) * h)) ** 2
        )

        closest_link = links_in_cell.loc[
            links_in_cell["dist_to_cell_center"].idxmin(),
            "link_idx"
        ]

        starting_links.append(closest_link)

    return starting_links