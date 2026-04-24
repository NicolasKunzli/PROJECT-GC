"""
clustering/ward.py — Agglomerative (Ward) clustering of road segments.

Ward linkage with a network-connectivity constraint ensures every resulting
cluster is a contiguous subgraph of the road network. Because clusters must be
spatially connected, this method is complementary to unconstrained KMeans.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

from config import links, NETWORK_CONNECTIVITY
from network.draw import sublink, polyg
from clustering.features import build_cluster_features


def clustering(n_clusters, name, feature_type, threshold=np.array([]), session_min=0, session_max=100):
    """
    Ward-linkage agglomerative clustering of road links with adjacency constraint.

    Every cluster is guaranteed to form a contiguous subgraph because the
    sklearn connectivity matrix limits merges to neighbouring links only.

    Parameters
    ----------
    n_clusters   : int – desired number of clusters
    name         : str – subfolder / filename prefix under figure/clustering/
    feature_type : str – one of {"geometric", "speed", "distance", "time"}
    threshold    : int ndarray – low-speed / short-link indices (highlighted in black)
    session_min, session_max : int – session slice for mean_over_sessions
    """
    folder = f"figure/clustering/{name}"
    os.makedirs(folder, exist_ok=True)

    X = build_cluster_features(feature_type, session_min=session_min, session_max=session_max)

    if feature_type == "speed" and threshold.size > 0:
        print(f"The low speed and short links are: {threshold}")

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=NETWORK_CONNECTIVITY,
    ).fit_predict(X)

    plot_links            = links.copy()
    plot_links["cluster"] = labels
    cluster_sizes         = np.bincount(labels, minlength=n_clusters)
    print(f"{name}: connected Ward clusters, size range {cluster_sizes.min()}-{cluster_sizes.max()}")

    cmap_discrete  = matplotlib.colormaps.get_cmap("tab10").resampled(n_clusters)
    cluster_colors = cmap_discrete(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(dpi=250)
    ax.set_aspect("equal")
    ax.set_title(f"{name} (connected Ward, k={n_clusters})", fontsize=9)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    for idx, row in plot_links.iterrows():
        x, y  = sublink(row)
        color = cluster_colors[int(row["cluster"])]
        z     = 1
        if idx in threshold:
            color = "black"
            z     = 3
        ax.plot(x, y, c=color, linewidth=0.4 + row["num_lanes"] * 0.4, zorder=z)

    handles = [
        plt.Line2D([0], [0], color=cluster_colors[k], lw=3, label=f"Cluster {k}")
        for k in range(n_clusters)
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    out = f"{folder}/{name}_best.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")
