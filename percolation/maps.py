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
from percolation.core import compute_normalized_speed
from processing.speed import fill_speed_nans, mean_over_sessions


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
