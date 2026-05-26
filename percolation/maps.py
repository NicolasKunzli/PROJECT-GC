"""
percolation/maps.py — Congestion visualisation built on percolation results.

congestion_map        : per-timestep maps showing red/green congestion status.
grid_clust            : rectangular grid overlay with per-cell congestion colour.
grid_clust_kmeans     : K-means partition on (cell_x, cell_y, avg_speed) grid.
grid_clust_kmeans_temporal : temporal median-speed plot for fixed K-means clusters.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import DL, links, LOCAL_FIGURE, NODATA_COLOR
from network.draw import sublink, polyg
from percolation.core import compute_normalized_speed
from processing.speed import fill_speed_nans, mean_over_sessions

_CLUSTER_PALETTE = ["#e6194b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
                    "#bfef45", "#f032e6", "#469990", "#9A6324", "#800000"]


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


def _fit_kmeans_on_grid(xdiv, ydiv, qc, session_min=0, session_max=100, percentile=65):
    """
    Shared setup for grid K-means: compute cell-average speed, run K-means.

    Returns
    -------
    links_local     : DataFrame with cell_x, cell_y, speed_mean, nodata columns
    cells           : DataFrame with cell_x, cell_y, cell_avg_speed, cluster columns
    cluster_mean_spd: Series – mean speed per cluster
    cluster_is_cong : Series – bool, True if cluster mean speed < threshold
    cluster_color   : dict   – cluster id → hex colour
    threshold       : float  – congestion threshold used
    x_min, x_max, y_min, y_max, w, h : grid geometry
    nodata_mask     : bool ndarray (N,)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    links_local = links.copy()
    tol   = 100
    x_min = np.min(links_local["from_x"]) - tol
    x_max = np.max(links_local["to_x"])   + tol
    y_min = np.min(links_local["from_y"]) - tol
    y_max = np.max(links_local["to_y"])   + tol

    w = (x_max - x_min) / xdiv
    h = (y_max - y_min) / ydiv

    links_local["cell_x"] = np.clip(
        ((links_local["c_x"] - x_min) // w).astype(int), 0, xdiv - 1)
    links_local["cell_y"] = np.clip(
        ((links_local["c_y"] - y_min) // h).astype(int), 0, ydiv - 1)

    if qc is not None:
        r = compute_normalized_speed()
        r_profile, nodata_mask = fill_speed_nans(
            mean_over_sessions(r, min=0, max=r.shape[0])
        )
        links_local["speed_mean"] = np.nanmean(r_profile, axis=0)
        links_local["nodata"]     = nodata_mask
        threshold = qc
    else:
        vdist = DL._vdist_3min.astype(float)
        vtime = DL._vtime_3min.astype(float)
        speed = np.divide(vdist, vtime,
                          out=np.full(vdist.shape, np.nan),
                          where=(vtime != 0) & (~np.isnan(vtime)))
        sp, nodata_mask = fill_speed_nans(
            mean_over_sessions(speed, min=session_min, max=session_max))
        avg = np.nanmean(sp, axis=0)
        links_local["speed_mean"] = avg
        links_local["nodata"]     = nodata_mask
        threshold = np.nanpercentile(avg, percentile)

    cell_speed = (links_local.groupby(["cell_x", "cell_y"])["speed_mean"]
                  .mean().reset_index(name="cell_avg_speed"))
    cells = cell_speed.dropna(subset=["cell_avg_speed"]).copy()

    X = StandardScaler().fit_transform(
        cells[["cell_x", "cell_y", "cell_avg_speed"]].values.astype(float)
    )
    return (links_local, cells, X, threshold, nodata_mask,
            x_min, x_max, y_min, y_max, w, h)


def grid_clust_kmeans(xdiv=20, ydiv=16, k=5, qc=None, percentile=65,
                      session_min=0, session_max=100):
    """
    Grid clustering with K-means (k clusters) on (cell_x, cell_y, avg_speed).

    Each cluster is labelled congested (mean speed < threshold) or functional.
    Congested cluster cells are hatched; each cluster gets a distinct colour.
    Saves a spatial map figure.

    Parameters
    ----------
    xdiv, ydiv  : int   – grid divisions
    k           : int   – number of K-means clusters
    qc          : float or None – normalised percolation threshold (None → percentile mode)
    percentile  : int   – speed percentile threshold (ignored when qc is set)
    session_min, session_max : int – session slice (percentile mode only)
    """
    from sklearn.cluster import KMeans
    import matplotlib.patches as mpatches

    (links_local, cells, X, threshold, nodata_mask,
     x_min, x_max, y_min, y_max, w, h) = _fit_kmeans_on_grid(
        xdiv, ydiv, qc, session_min, session_max, percentile)

    xs = np.arange(x_min, x_max, w)
    ys = np.arange(y_min, y_max, h)

    folder = "figure/clustering/grid_clusters"
    os.makedirs(folder, exist_ok=True)

    n_clusters = min(k, len(cells))
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cells["cluster"] = km.fit_predict(X)

    cluster_mean_spd  = cells.groupby("cluster")["cell_avg_speed"].mean()
    cluster_is_cong   = cluster_mean_spd < threshold

    cluster_color = {
        c: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        for i, c in enumerate(sorted(cells["cluster"].unique()))
    }

    links_local = links_local.merge(
        cells[["cell_x", "cell_y", "cluster"]], on=["cell_x", "cell_y"], how="left"
    )
    links_local["color"] = (
        links_local["cluster"].map(cluster_color).fillna(NODATA_COLOR)
    )
    links_local.loc[links_local["nodata"], "color"] = NODATA_COLOR

    fig, ax = plt.subplots(dpi=250)

    for _, row in links_local.iterrows():
        ax.plot([row["from_x"], row["to_x"]], [row["from_y"], row["to_y"]],
                color=row["color"], linewidth=0.8)

    cell_lookup = cells.set_index(["cell_x", "cell_y"])
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            key = (xi, yi)
            if key in cell_lookup.index:
                cell_row = cell_lookup.loc[key]
                clr   = cluster_color[cell_row["cluster"]]
                hatch = "///" if cluster_is_cong[cell_row["cluster"]] else ""
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
            hatch="///" if cluster_is_cong[c] else "",
            label=f"Cluster {c + 1} — {'congested' if cluster_is_cong[c] else 'functional'}",
        )
        for c in sorted(cells["cluster"].unique())
    ]
    ax.legend(handles=handles, fontsize=6, loc="best")

    polyg(ax, color="black", zorder=-2)
    if qc is not None:
        title_mode = f"$q_c$={qc:.3f}"
        fname = f"{folder}/grid_kmeans{k}_qc{qc:.3f}.png"
    else:
        title_mode = f"{percentile}th pct"
        fname = f"{folder}/grid_kmeans{k}_{percentile}pct.png"
    ax.set_title(f"Grid K-means k={k}, {title_mode}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    fig.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
    return fname


def grid_clust_kmeans_temporal(xdiv=20, ydiv=16, k=5, qc=None, percentile=65,
                               session_min=0, session_max=100,
                               timesteps=None):
    """
    Fit K-means on time-averaged grid (same as grid_clust_kmeans), then plot
    median raw speed per cluster over the given representative timesteps.

    Saves a line-plot figure alongside the spatial map.

    Parameters
    ----------
    timesteps : list of int or None – indices into the 39-step analysis window.
                Defaults to the 15 standard report timesteps.
    """
    from sklearn.cluster import KMeans

    if timesteps is None:
        timesteps = [0, 3, 5, 7, 9, 13, 17, 19, 21, 23, 25, 29, 33, 36, 38]

    (links_local, cells, X, threshold, nodata_mask,
     x_min, x_max, y_min, y_max, w, h) = _fit_kmeans_on_grid(
        xdiv, ydiv, qc, session_min, session_max, percentile)

    folder = "figure/clustering/grid_clusters"
    os.makedirs(folder, exist_ok=True)

    n_clusters = min(k, len(cells))
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cells["cluster"] = km.fit_predict(X)

    cluster_mean_spd = cells.groupby("cluster")["cell_avg_speed"].mean()
    cluster_is_cong  = cluster_mean_spd < threshold

    # Merge cluster labels into links_local
    links_local = links_local.merge(
        cells[["cell_x", "cell_y", "cluster"]], on=["cell_x", "cell_y"], how="left"
    )

    # ── Load raw speed for all sessions, average, fill NaNs ──────────────────
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)
    raw_speed_all = np.divide(
        vdist, vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=(vtime != 0) & (~np.isnan(vtime)),
    )
    raw_profile, _ = fill_speed_nans(
        mean_over_sessions(raw_speed_all, min=0, max=raw_speed_all.shape[0])
    )  # shape (T, N)

    # ── For each timestep, compute median raw speed per cluster ───────────────
    cluster_ids = sorted(cells["cluster"].unique())
    n_t = len(timesteps)

    # Map link index → cluster id (-1 if nodata or unassigned)
    link_cluster = links_local["cluster"].values  # float (NaN if not assigned)

    medians = np.full((n_t, len(cluster_ids)), np.nan)

    for ti, t in enumerate(timesteps):
        raw_t = raw_profile[t]  # (N,)
        for ci, cid in enumerate(cluster_ids):
            mask = (link_cluster == cid) & ~nodata_mask & ~np.isnan(raw_t)
            if mask.sum() > 0:
                medians[ti, ci] = np.median(raw_t[mask])

    # ── Time labels (HH:MM) ───────────────────────────────────────────────────
    def _t_label(t):
        total_min = 8 * 60 + 3 + t * 3
        return f"{total_min // 60:02d}:{total_min % 60:02d}"

    x_labels = [_t_label(t) for t in timesteps]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    cluster_mean_spd_sorted = cluster_mean_spd.sort_values()
    rank_order = list(cluster_mean_spd_sorted.index)  # slowest first

    for rank, cid in enumerate(rank_order):
        ci = cluster_ids.index(cid)
        cong_label = "congested" if cluster_is_cong[cid] else "functional"
        ax.plot(
            range(n_t), medians[:, ci],
            label=f"Cluster {cid + 1} ({cong_label})",
            color=_CLUSTER_PALETTE[rank % len(_CLUSTER_PALETTE)],
            marker="o", markersize=4, linewidth=1.5,
        )

    ax.set_xticks(range(n_t))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel("Median raw speed (m/s)", fontsize=9)
    ax.axhline(threshold * (13.889 if qc is not None else 1),
               color="gray", linestyle="--", linewidth=0.8,
               label=f"Threshold ≈ {threshold:.3f}" + (" (norm.)" if qc is not None else " m/s"))
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"Grid K-means k={k} — median raw speed per cluster over time\n"
        f"({'$q_c$=' + str(qc) if qc else str(percentile) + 'th pct'})",
        fontsize=9,
    )

    fig.tight_layout()
    if qc is not None:
        fname = f"{folder}/grid_kmeans{k}_qc{qc:.3f}_median.png"
    else:
        fname = f"{folder}/grid_kmeans{k}_{percentile}pct_median.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
    return fname, medians, cluster_ids, cluster_is_cong
