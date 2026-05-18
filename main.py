"""

Module map

  config              — paths, constants, DataLoader singleton (DL), metadata
  processing.speed    — mean_over_sessions, fill_speed_nans, NaN policy
  network.draw        — sublink, link, polyg, graph (primitive map drawing)
  network.simplify    — group_segments, simplified_map
  clustering.features — build_cluster_features, thresholds
  clustering.kmeans   — kmeans_clustering
  clustering.ward     — clustering (AgglomerativeClustering / Ward)
  percolation.core    — percolation_analysis, find_bottlenecks
  percolation.maps    — congestion_map, grid_clust
  gif                 — gradient_gif
  utils               — closest_link

"""

import os
import numpy as np

from config import DL, links

from network.draw     import graph
from network.simplify import simplified_map

from clustering.features import thresholds
from clustering.kmeans   import kmeans_clustering, plot_cluster_speed, run_kmeans_graph
from clustering.ward     import clustering

from percolation.core import percolation_analysis
from percolation.maps import compare_congestion_methods, congestion_map, grid_clust, grid_clust_kmeans

from gif   import gradient_gif
from utils import closest_link

import matplotlib.pyplot as plt

# ── colors ──────────────────────────────────────────────────────────
global_cmap = plt.get_cmap("tab10")

cluster_colors = {
    k: global_cmap(np.linspace(0, 1, k, endpoint=False))
    for k in range(1, 11)
}

if __name__ == "__main__":

    os.makedirs("figure/clustering", exist_ok=True)

    # ── Baseline map ──────────────────────────────────────────────────────────
    # graph()

    # ── GIF parameters ────────────────────────────────────────────────────────
    fps = 0.5
    param = [
        DL._vdist_3min,
        DL._vtime_3min,
        DL._vdist_3min / DL._vtime_3min,
    ]
    param_name = [
        "vdist_3min",
        "vtime_3min",
        "vdist_3min_over_vtime_3min",
    ]
    # gradient_gif(param, param_name, fps)

    # ── Grid clustering ───────────────────────────────────────────────────────
    n_clus = 8
    # grid_clust(20, 16, 85)
    # grid_clust(20, 16, 65)
    # grid_clust(20, 16, 55)
    # grid_clust(20, 16, 45)

    # ── Threshold: low-speed and short links ──────────────────────────────────
    th = thresholds(max_speed=2, max_length=np.inf)

    # ── Ward (agglomerative) — feature-driven, connectivity-constrained ───────
    # clustering(n_clus, "geometric_clusters", "geometric")
    # clustering(n_clus, "distance_clusters",  "distance")
    # clustering(n_clus, "time_clusters",      "time")
    # clustering(n_clus, "speed_clusters",     "speed", threshold=th)

    # ── KMeans — traffic features, no connectivity constraint ─────────────────
    # kmeans_clustering(n_clus, "kmeans_speed_clusters", "speed", threshold=th)
    # kmeans_clustering(n_clus, "kmeans_speed_clusters", "speed", threshold=th, filter=True)
    # kmeans_clustering(n_clus, "kmeans_distance_clusters", "distance")
    # kmeans_clustering(n_clus, "kmeans_time_clusters",     "time")

    # ── Simplified map ────────────────────────────────────────────────────────
    # simplified_map(distance_threshold=50.0, grad=True)
    # simplified_map(distance_threshold=50.0, grad=False, color="navy")

    # # ── Percolation analysis ──────────────────────────────────────────────────
    # qc, bottlenecks = percolation_analysis(session=0)
    # congestion_map(qc, session=0)
    # grid_clust(20, 16, qc=qc)  # grid coloured by percolation threshold
    # grid_clust_kmeans(20, 16, k=5, qc=qc)  # 5 K-means clusters on (x, y, speed)
    # compare_congestion_methods(
    #     session=0,
    #     timestep=31,
    #     percentile=30,
    #     grid_percentile=50,
    #     xdiv=20,
    #     ydiv=16,
    # )

    # ── Spawn-point selection ─────────────────────────────────────────────────
    # Pre-compute network bounding-box fractile coordinates
    x_min_lim = np.min(links["from_x"])
    x_max_lim = np.max(links["to_x"])
    y_min_lim = np.min(links["from_y"])
    y_max_lim = np.max(links["to_y"])

    x25 = x_min_lim + 0.25 * (x_max_lim - x_min_lim)
    x50 = x_min_lim + 0.50 * (x_max_lim - x_min_lim)
    x75 = x_min_lim + 0.75 * (x_max_lim - x_min_lim)

    y25 = y_min_lim + 0.25 * (y_max_lim - y_min_lim)
    y50 = y_min_lim + 0.50 * (y_max_lim - y_min_lim)
    y75 = y_min_lim + 0.75 * (y_max_lim - y_min_lim)

    # Three spawn configurations (2×2, 1×2, 2×1)
    links_spawn_1 = [(x, y) for x in [x25, x75] for y in [y25, y75]]
    links_spawn_2 = [(x, y) for x in [x50]       for y in [y25, y75]]
    links_spawn_3 = [(x, y) for x in [x25, x75] for y in [y50]]

    init_links_1 = [closest_link(x, y) for x, y in links_spawn_1]
    init_links_2 = [closest_link(x, y) for x, y in links_spawn_2]
    init_links_3 = [closest_link(x, y) for x, y in links_spawn_3]

    cluster_amount = list(range(2, 8))

    # Timeframes covering the start (≈08:18), middle (≈09:03), and end (≈09:48) of the window
    # timeframes = [5, 20, 35]
    # for i in cluster_amount:
    #     kmeans_clustering(i, "kmeans_speed_clusters", "speed")
    #     for tf in timeframes:
    #         kmeans_clustering(i, f"kmeans_speed_clusters_t_{tf}", "speed",
    #                           dynamic_weight=1.5, spatial_weight=2.0, timeframe=tf)

    # kmeans_clustering(5, "kmeans_speed_clusters_t_10_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=10, init_links=init_links_1)
    # kmeans_clustering(len(init_links_2), "kmeans_speed_clusters_t_10_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=10, init_links=init_links_2)
    # kmeans_clustering(len(init_links_3), "kmeans_speed_clusters_t_10_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=10, init_links=init_links_3)

    # kmeans_clustering(len(init_links_1), "kmeans_speed_clusters_t_30_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=30, init_links=init_links_1)
    # kmeans_clustering(len(init_links_2), "kmeans_speed_clusters_t_30_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=30, init_links=init_links_2)
    # kmeans_clustering(len(init_links_3), "kmeans_speed_clusters_t_30_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=30, init_links=init_links_3)

    # main_roads1 = [
    #     closest_link(429250, 4581750),
    #     closest_link(432200, 4581900),
    #     closest_link(430280, 4582250),
    # ]
    # kmeans_clustering(len(main_roads1), "kmeans_speed_clusters_t_30_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=30, init_links=main_roads1)

    # main_roads2 = [
    #     closest_link(429250, 4581750),
    #     closest_link(431250, 4582350),
    # ]
    # kmeans_clustering(len(main_roads2), "kmeans_speed_clusters_t_30_set_spawn", "speed",
    #                   dynamic_weight=1.5, spatial_weight=2.0, timeframe=30, init_links=main_roads2)

    # ── Lowest / highest speed weight analysis ────────────────────────────────
    # Mean speed over all sessions (used for identifying representative slow/fast links)
    # speed = np.divide(
    #     DL._vdist_3min.astype(float),
    #     DL._vtime_3min.astype(float),
    #     where=DL._vtime_3min.astype(float) != 0,
    # )
    # from processing.speed import mean_over_sessions
    # speed = mean_over_sessions(speed)
    # speed = np.mean(speed, axis=0)

    # df = links.copy()
    # df["speed"] = speed
    # df = df[(df["speed"].notna()) & (df["speed"] != 0)]
    # df_sorted = df.sort_values("speed")

    # low_speeds  = df_sorted.head(20)["speed"]
    # high_speeds = df_sorted.tail(20).iloc[::-1]["speed"]

    # idx_low  = [2, 7, 11, 19]
    # idx_high = [3, 10, 15]
    # combinations1 = (
    #     low_speeds.iloc[idx_low].index.tolist()
    #     + high_speeds.iloc[idx_high].index.tolist()
    # )

    timesteps = [0, 1, 3, 5, 7, 9, 
                 13, 
                 17, 19, 21, 23, 25, 
                 29, 
                 33, 36, 38
                 ]

    bottlenecks1 = [
        int(closest_link(431900, 4582650)),  # upper right
        int(closest_link(430570, 4582390)),  # upper middle
        int(closest_link(428730, 4582880)),  # upper left
        int(closest_link(428700, 4581420)),  # lower left
        int(closest_link(431450, 4581820)),  # middle right
        int(closest_link(432200, 4581370)),  # lower right
        int(closest_link(430000, 4581800)),  # lower middle
        int(closest_link(429060, 4582220)),  # middle left
    ]
    
    mixed1 = [#bottlenecks
        int(closest_link(431900, 4582650)),  # upper right
        int(closest_link(430570, 4582390)),  # upper middle
        int(closest_link(428730, 4582880)),  # upper left
        int(closest_link(428700, 4581420)),  # lower left
        int(closest_link(431450, 4581820)),  # middle right
        int(closest_link(432200, 4581370)),  # lower right
        int(closest_link(430000, 4581800)),  # lower middle
        int(closest_link(429060, 4582220)),  # middle left
        #not bottlenecks
        int(closest_link(429800, 4582800)),  # middle top
    ]
    
    mixed2 = [#bottlenecks
        int(closest_link(431900, 4582650)),  # upper right
        int(closest_link(430570, 4582390)),  # upper middle
        int(closest_link(428730, 4582880)),  # upper left
        int(closest_link(428700, 4581420)),  # lower left
        int(closest_link(431450, 4581820)),  # middle right
        int(closest_link(432200, 4581370)),  # lower right
        int(closest_link(430000, 4581800)),  # lower middle
        int(closest_link(429060, 4582220)),  # middle left
        #not bottlenecks
        int(closest_link(428400, 4582300)),  # middle lefter
    ]        

     
    
#     # Spatial weight sweep
#     dyn_weights = np.array(list(range(1, 21))) / 10
#     for i in dyn_weights:
#         kmeans_clustering(len(bottlenecks1), "kmeans_speed_clusters_t_31_bottlenecks_weights",
#                           "speed", dynamic_weight=1, spatial_weight=i,
#                           timeframe=31, init_links=bottlenecks1, show_weights=True)

    # Session-stratified clustering
    sessions = [
        (0, 0),
        (1, 20),
        (21, 40),
        (41, 60),
        (61, 80),
        (81, 100) # 81-100 Empty ?
        ]
    # 7 Clusters no lr
    bottlenecks_no_lr = [
        b for i, b in enumerate(bottlenecks1)
        if i != 5  # lower right
    ]
    
    # 7 Clusters no lower right
    mean_speed_time = run_kmeans_graph(
        n_clusters=len(bottlenecks_no_lr),
        name="kmeans_speed_clusters_bottlenecks/7-no_lr",
        init_links=bottlenecks_no_lr,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(bottlenecks_no_lr)],
        session_min=0,
        session_max=100
    )        
        
    # 7 Clusters no middle left
    mean_speed_time = run_kmeans_graph(
        n_clusters=len(bottlenecks1[:-1]),
        name="kmeans_speed_clusters_bottlenecks/7-no_ml",
        init_links=bottlenecks1[:-1],
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(bottlenecks1[:-1])],
        session_min=0,
        session_max=100
    )
        
    # 8 Clusters
    mean_speed_time = run_kmeans_graph(
        n_clusters=len(bottlenecks1),
        name="kmeans_speed_clusters_bottlenecks/8",
        init_links=bottlenecks1,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(bottlenecks1)],
        session_min=0,
        session_max=100
    )
    
    # 9 mixed2, middle left
    run_kmeans_graph(
        n_clusters=len(mixed2),
        name="kmeans_speed_clusters_bottlenecks/9-middle-left",
        init_links=mixed2,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(mixed2)],
        session_min=0,
        session_max=100
    )  
    
    # 9 mixed1, middle up
    run_kmeans_graph(
        n_clusters=len(mixed1),
        name="kmeans_speed_clusters_bottlenecks/9-middle-up",
        init_links=mixed1,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(mixed1)],
        session_min=0,
        session_max=100
    )



### Graphs for sessions     

for s_min, s_max in sessions:
    
    # 7 No lower right
    run_kmeans_graph(
        n_clusters=len(bottlenecks_no_lr),
        name="kmeans_speed_clusters_bottlenecks_sessions/7-no_lr",
        init_links=bottlenecks_no_lr,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(bottlenecks_no_lr)],
        session_min=s_min,
        session_max=s_max
    )
    
    # 7 Clusters no middle left
    mean_speed_time = run_kmeans_graph(
        n_clusters=len(bottlenecks1[:-1]),
        name="kmeans_speed_clusters_bottlenecks_sessions/7-no_ml",
        init_links=bottlenecks1[:-1],
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(bottlenecks1[:-1])],
        session_min=s_min,
        session_max=s_max
    )
    
    # 8 bottlenecks1
    run_kmeans_graph(
        n_clusters=len(bottlenecks1),
        name="kmeans_speed_clusters_bottlenecks_sessions/8",
        init_links=bottlenecks1,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(bottlenecks1)],
        session_min=s_min,
        session_max=s_max
    )

    # 9 mixed2, middle left
    run_kmeans_graph(
        n_clusters=len(mixed2),
        name="kmeans_speed_clusters_bottlenecks_sessions/9-middle-left",
        init_links=mixed2,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(mixed2)],
        session_min=s_min,
        session_max=s_max
    )    

    # 9 mixed1, middle up
    run_kmeans_graph(
        n_clusters=len(mixed1),
        name="kmeans_speed_clusters_bottlenecks_sessions/9-middle-up",
        init_links=mixed1,
        timeframe=timesteps,
        cluster_colors=cluster_colors[len(mixed1)],
        session_min=s_min,
        session_max=s_max
    )

       

        

