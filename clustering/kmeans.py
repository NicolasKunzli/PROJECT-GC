"""
clustering/kmeans.py — KMeans clustering of road segments by traffic behaviour.

Unlike Ward (which enforces network adjacency), KMeans groups links with similar
speed/distance/time profiles regardless of their spatial position. Cluster spread
is controlled by the spatial_weight parameter in the feature builder.

HOW KMEANS WORKS — THE FOUR STEPS
───────────────────────────────────────────────────────────────────────────────
KMeans partitions N data points into k groups by minimising total within-cluster
variance (inertia). Each iteration has four conceptual steps:

  STEP 1 – INITIALISE CENTROIDS
      Place k centroids. Default strategy (k-means++) spreads them far apart,
      reducing the chance of a bad local minimum. `n_init="auto"` restarts
      several times and keeps the lowest-inertia result.

  STEP 2 – LABEL EACH DATA POINT
      Assign every link to its nearest centroid using Euclidean distance in
      feature space — "nearest" means "most similar traffic behaviour".

  STEP 3 – UPDATE CENTROIDS
      Recompute each centroid as the mean of its assigned links.

  STEP 4 – REPEAT UNTIL CONVERGENCE
      Repeat steps 2–3 until centroids shift < tol (1e-4) or max_iter (300).
      All four steps run inside `.fit_predict(X)`.
───────────────────────────────────────────────────────────────────────────────
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from config import DL, links
from network.draw import sublink, polyg
from clustering.features import build_cluster_features
from processing.speed import mean_over_sessions, fill_speed_nans
from utils import closest_link



def kmeans_clustering(
    n_clusters,
    name,
    feature_type,
    spatial_weight=1.75,
    dynamic_weight=1,
    random_state=42,
    threshold=np.array([]),
    filter=False, # WARNING : Filtering may mismatch indices, BUGFIX NEEDED
    timeframe=None,
    init_links=None,
    show_weights=False,
    session_min=0,
    session_max=100,
    spatial_centroids=False,
    colors=None
):
    """
    Cluster road links with KMeans and save a colour-coded network map.

    Parameters
    ----------
    n_clusters    : int – number of clusters (overridden by len(init_links) if provided)
    name          : str – subfolder / filename prefix under figure/clustering/
    feature_type  : str – one of {"geometric", "speed", "distance", "time"}
    spatial_weight : float – scale factor for spatial coordinates in the feature matrix
    dynamic_weight : float – scale factor for temporal features in the feature matrix
    random_state  : int – reproducibility seed
    threshold     : int ndarray – indices of low-speed / short links
    filter        : bool – drop threshold segments before clustering
    timeframe     : int or None – if set, uses a single-timestep feature snapshot
    init_links    : list or None – force specific links as initial cluster centroids
    show_weights  : bool – append spatial/dynamic weight values to the output filename
    session_min, session_max : int – session slice for averaging, session_max is included
    """
    folder = f"figure/clustering/{name}"
    os.makedirs(folder, exist_ok=True)
    
    
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)
    
    speed = np.divide(
        vdist, vtime,
        out=np.full(vdist.shape, np.nan),
        where=vtime != 0
    )
    
    speed_sel = speed[session_min:session_max + 1, timeframe, :] # Sessions
    nan_ratio = np.mean(np.isnan(speed_sel), axis=(0))  # (link_idx,)
    bad_links = np.where(nan_ratio > 0.3)[0]
    bad_links_set = set(bad_links)
    threshold_set = set(threshold)
    # ── Build feature matrix X  (N_links × n_features) ────────────────────────
    one_tf = timeframe is not None
    X = build_cluster_features(
        feature_type,
        spatial_weight=spatial_weight,
        dynamic_weight=dynamic_weight,
        timeframe=timeframe if one_tf else 0,
        threshold=threshold,
        filter=filter,
        one_timeframe=one_tf,
        session_min=session_min,
        session_max=session_max,
    )

    if feature_type == "speed" and threshold.size > 0:
        print(f"The low speed and short links are: {threshold}")

    # ── Fit KMeans ─────────────────────────────────────────────────────────────
    spawn_links = []

    if init_links is not None:
        if len(init_links) != int(n_clusters):
            print(f"n_clusters doesn't match the amount of init_links")
            n_clusters = len(init_links)
            print(f"n_clusters set to {len(init_links)}")
        init_centroids = X[init_links]
        n_clusters     = len(init_links)
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_centroids,
            n_init=1,
            random_state=random_state,
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )

    labels    = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Find the link closest to each centroid (the cluster "spawn point")
    if init_links is None:
        for k in range(n_clusters):
            cluster_idx   = np.where(labels == k)[0]
            cluster_points = X[cluster_idx]
            distances      = np.linalg.norm(cluster_points - centroids[k], axis=1)
            spawn_links.append(cluster_idx[np.argmin(distances)])

    # ── Prepare plot DataFrame ─────────────────────────────────────────────────
    plot_links           = links.copy()
    if filter and threshold.size > 0:
        mask           = np.ones(plot_links.shape[0], dtype=bool)
        mask[threshold] = False
        plot_links     = plot_links.loc[mask]
    plot_links["cluster"] = labels

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    print(f"{name}: KMeans clusters, size range {cluster_sizes.min()}-{cluster_sizes.max()}")

    # ── Colour palette ─────────────────────────────────────────────────────────
    if colors is None:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        cluster_colors = cmap(np.linspace(0, 1, n_clusters))
    else:
        cluster_colors = np.asarray(colors)
    if cluster_colors.shape[0] != n_clusters:
        raise ValueError(
            f"colors size ({cluster_colors.shape[0]}) != n_clusters ({n_clusters})"
        )
    fig, ax = plt.subplots(dpi=250)
    ax.set_aspect("equal")
    title = f"{name}_filtered (KMeans, k={n_clusters})" if filter else f"{name} (KMeans, k={n_clusters})"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    # ── Draw links coloured by cluster ─────────────────────────────────────────
    for idx, row in plot_links.iterrows():
        x, y = sublink(row)

        z = 1
        color = cluster_colors[int(row["cluster"])]

        if idx in bad_links_set:
            color = "black"
            z = 3

        elif idx in threshold_set:
            color = "black"
            z = 3

        ax.plot(
            x, y,
            c=color,
            linewidth=0.4 + row["num_lanes"] * 0.4,
            zorder=z
        )

    # ── Highlight spawn / seed links ───────────────────────────────────────────
    all_spawn = list(spawn_links) + (list(init_links) if init_links is not None else [])
    for idx in set(all_spawn):
        row  = plot_links.iloc[idx]
        x, y = sublink(row)
        ax.plot(x, y, c="lime", linewidth=2 + row["num_lanes"] * 0.4, zorder=4)

    # ── Draw filtered-out (threshold) links in black ───────────────────────────
    if filter:
        for idx, row in links.loc[threshold].iterrows():
            x, y = sublink(row)
            ax.plot(x, y, c="black", linewidth=0.4 + row["num_lanes"] * 0.4, zorder=3)

    polyg(ax, color="black", alpha=0.6, zorder=-1)

    # ── Legend: cluster index + mean speed ────────────────────────────────────
    cluster_mean_speeds = []

    for k in range(n_clusters):
        cluster_idx = np.where(labels == k)[0]

        vdist_k = vdist[session_min:session_max+1, timeframe, cluster_idx]
        vtime_k = vtime[session_min:session_max+1, timeframe, cluster_idx]

        total_dist = np.nansum(vdist_k)
        total_time = np.nansum(vtime_k)

        mean_cluster_speed = total_dist / total_time if total_time > 0 else np.nan
        cluster_mean_speeds.append(mean_cluster_speed)

    handles = [
        plt.Line2D(
            [0], [0],
            color=cluster_colors[k],
            lw=3,
            label=f"Cluster {k} – {cluster_mean_speeds[k]:.2f} m/s"
        )
        for k in range(n_clusters)
    ]

    handles.append(
        plt.Line2D([0], [0], color="lime", lw=3, label="Spawn points")
    )

    ax.legend(handles=handles, fontsize=5, loc="upper right")

    # ── Build output filename ──────────────────────────────────────────────────
    all_sessions = session_min == 0 and session_max == 100
    session_str = "" if all_sessions else f"s{session_min}-{session_max}"

    # ── Normalize path ─────────────────────────────────────────────
    name = name.replace("\\", "/").strip("/")
    name_parts = name.split("/")
    principal_name = name_parts[0]

    # ── Folder structure ───────────────────────────────────────────────────────
    full_folder = os.path.join(folder, session_str)
    os.makedirs(full_folder, exist_ok=True)

    # ── Base filename ──────────────────────────────────────────────────────────
    base_parts = [str(n_clusters), f"t{timeframe}"]
    if session_str:
        base_parts.append(session_str)
    base_parts.append(principal_name)
    base = "_".join(base_parts)

    # ── Suffix ────────────────────────────────────────────────────────────────
    if filter:
        suffix = "filtered"
    elif init_links is not None:
        suffix = "_".join(str(int(x)) for x in init_links)
    else:
        suffix = "best"

    # ── FINAL FILE ─────────────────────────────────────────────────────────────
    filename = os.path.join(full_folder, f"{base}_{suffix}.png")

    if show_weights:
        filename = filename.replace(
            ".png",
            f"_spa{spatial_weight}_dyn{dynamic_weight}.png"
        )

    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved → {filename}")

    # ── Return spatial centroid ──────────────────────────────────────────────────
    if spatial_centroids:

        # ── 1. Compute spatial centroids ─────────────────────────────
        centroids_xy = []

        for k in range(n_clusters):
            cluster_idx = np.where(labels == k)[0]
            
            xy = []
            for idx in cluster_idx:
                row = links.iloc[idx]
                x, y = sublink(row)
                xy.append((np.mean(x), np.mean(y)))
            
            xy = np.array(xy)
            cx = xy[:, 0].mean()
            cy = xy[:, 1].mean()
            
            centroids_xy.append((cx, cy))

        # ── 2. Find closest link to each spatial centroid ────────────
        spawn_links_xy = [
            closest_link(cx, cy, np.where(labels == k)[0])
            for k, (cx, cy) in enumerate(centroids_xy)
        ]

        # ── 3. Return links ──────────────────────────────────────────
        print(spawn_links_xy)
        return spawn_links_xy
    return cluster_mean_speeds, cluster_colors, folder


def plot_cluster_speed(
    timesteps,
    values,
    name,
    folder,
    colors=None,
):
    """
    Plot already computed cluster speeds on a graph with time on the x-axis and speed on the y-axis.

    Parameters
    ----------
    timesteps       : array-like (T) – time values for each timestep
    values          : array-like (T, K) – mean speed per cluster over time
    name            : str – plot name
    folder          : str – output directory for saving the figure
    colors          : array-like (K, 4) or None – cluster colors (RGBA), optional
    """

    values = np.array(values)

    plt.figure()

    for k in range(values.shape[1]):

        plt.plot(
            timesteps,
            values[:, k],
            color=colors[k] if colors is not None else None,
            label=f"Cluster {k}"
        )

        plt.scatter(
            timesteps,
            values[:, k],
            color=colors[k] if colors is not None else None,
            s=15
        )

    # ── ensure folder exists ─────────────────────────────
    os.makedirs(folder, exist_ok=True)

    # ── sanitize name ─────────────────────────
    name = name.replace("\\", "/").strip("/")
    name_parts = name.split("/")
    safe_name = "_".join(name_parts)

    # ── simple filename ───────────────────────────────────
    filename = os.path.join(folder, f"{safe_name}.png")

    plt.xlabel("Time step")
    plt.ylabel("Mean speed (m/s)")
    plt.title(f"{name}")
    plt.legend(fontsize=6)
    plt.grid()

    plt.savefig(filename)
    plt.close()

    print(f"Saved → {filename}")
    
def run_kmeans_graph(
    n_clusters,
    name,
    init_links,
    timeframe,
    cluster_colors,
    session_min=0,
    session_max=100
):
    """
    Helper function combining the kmeans_clustering function and the plot_cluster_speed function.
    
    Parameters
    ----------
    n_clusters    : int – number of clusters (len(init_links) if not provided)
    name          : str – subfolder / filename prefix under figure/clustering/
    timeframe     : int or None – if set, uses a single-timestep feature snapshot
    init_links    : list or None – force specific links as initial cluster centroids
    session_min, session_max : int – session slice for averaging, session_max is included
    """

    mean_speed_time = None
    folder = None

    for t in timeframe:
        cluster_mean_speed, _, folder = kmeans_clustering(
            n_clusters,
            name,
            "speed",
            spatial_weight=2.0,
            dynamic_weight=1,
            timeframe=t,
            init_links=init_links,
            session_min=session_min,
            session_max=session_max,
            colors=cluster_colors,
            show_weights=True
        )

        if mean_speed_time is None:
            mean_speed_time = []

        mean_speed_time.append(cluster_mean_speed)

    mean_speed_time = np.array(mean_speed_time)

    plot_cluster_speed(
        timesteps=timeframe,
        values=mean_speed_time,
        name=f"{name}_s{session_min}-{session_max}",
        folder=folder,
        colors=cluster_colors
    )

    return mean_speed_time