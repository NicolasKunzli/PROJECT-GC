"""
percolation/maps.py — Congestion visualisation built on percolation results.

congestion_map  : per-timestep maps showing red/green congestion status.
grid_clust      : rectangular grid overlay with per-cell congestion colour.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def compare_congestion_methods(
    session=0,
    timestep=31,
    percentile=30,
    grid_percentile=50,
    xdiv=10,
    ydiv=8,
    qc=None,
    speed_threshold=None,
    grid_threshold=None,
    n_q=100,
    output=None,
):
    """
    Plot side-by-side congestion maps for direct threshold, grid aggregation,
    and percolation so the methods can be compared on the same snapshot.

    Parameters
    ----------
    session         : int        - simulation session index
    timestep        : int        - 3-minute timestep index
    percentile      : int/float  - raw-speed percentile used when speed_threshold is None
    grid_percentile : int/float  - raw-speed percentile used when grid_threshold is None
    xdiv, ydiv      : int        - grid divisions for the aggregation method
    qc              : float/None - percolation threshold; computed from the snapshot if None
    speed_threshold : float/None - raw-speed threshold in m/s; percentile is used if None
    grid_threshold  : float/None - grid threshold in m/s; grid_percentile is used if None
    n_q             : int        - q sweep resolution when qc is computed
    output          : str/None   - optional output path
    """
    raw_speed, raw_nodata = _raw_speed_snapshot(session, timestep)
    norm_speed, norm_nodata = _normalized_speed_snapshot(session, timestep)
    bounds = _network_bounds()

    raw_valid = ~(raw_nodata | np.isnan(raw_speed))
    if speed_threshold is None:
        speed_threshold = np.nanpercentile(raw_speed[raw_valid], percentile)
    if grid_threshold is None:
        grid_threshold = np.nanpercentile(raw_speed[raw_valid], grid_percentile)

    if qc is None:
        adj_directed = build_directed_adjacency()
        q_values = np.linspace(0, 1, n_q)
        qc, _, _ = find_critical_threshold(norm_speed, adj_directed, q_values)

    if output is None:
        folder = f"{LOCAL_FIGURE}/congestion_maps"
        os.makedirs(folder, exist_ok=True)
        output = f"{folder}/compare_methods_t{timestep}.png"
    else:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=250)

    _draw_segment_congestion(
        axes[0],
        raw_speed,
        raw_nodata,
        speed_threshold,
        f"Speed threshold\nv < {speed_threshold:.2f} m/s (p{percentile})",
        bounds,
    )
    _draw_grid_congestion(
        axes[1],
        raw_speed,
        raw_nodata,
        grid_threshold,
        xdiv,
        ydiv,
        f"Grid aggregation\n{xdiv}x{ydiv}, cell mean v < {grid_threshold:.2f} (p{grid_percentile})",
        bounds,
    )
    _draw_segment_congestion(
        axes[2],
        norm_speed,
        norm_nodata,
        qc,
        f"Percolation\nr < q_c={qc:.3f}",
        bounds,
    )

    handles = [
        plt.Line2D([0], [0], color=CONGESTED_COLOR, lw=3, label="Congested"),
        plt.Line2D([0], [0], color=FUNCTIONAL_COLOR, lw=3, label="Functional"),
        plt.Line2D([0], [0], color=NODATA_COLOR, lw=3, label="No data"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Congested areas comparison - session {session}, t={timestep} ({timestep * 3} min)", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.17, wspace=0.14)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved congestion-method comparison -> {output}")
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
