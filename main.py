import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
from DataLoad import DataLoader
import imageio
import re
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

############################# FILES/CLASS INSTANCES #############################
### Files
path = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
figure_path = os.path.join(path, "figure")
localfigure = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")
localgif = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")

os.makedirs(figure_path, exist_ok=True)
os.makedirs(localfigure, exist_ok=True)
os.makedirs(localgif, exist_ok=True)

centroid = pd.read_csv(os.path.join(path, "metadata", "centroid_pos.csv"))
links = pd.read_csv(os.path.join(path, "metadata", "link_bboxes.csv")).sort_values("id").reset_index(drop=True)
connections = pd.read_csv(os.path.join(path, "metadata", "connections.csv"))

polygons = pd.read_json(os.path.join(path, "metadata", "intersec_polygon.json"))
lane_info = pd.read_json(os.path.join(path, "metadata", "lane_info.json"))
od = pd.read_json(os.path.join(path, "metadata", "od_pairs.json"))

with open (os.path.join(path, "metadata", "train_test_split.json")) as f:tts=f.read()
with open(os.path.join(path, "metadata", "sections_of_interest.txt"), "r") as f:interest = [line.strip() for line in f.readlines()]


links["c_x"] = (links["from_x"] + links["to_x"])/2
links["c_y"] = (links["from_y"] + links["to_y"])/2


### Class
DL = DataLoader()
DL.init_graph_structure

############################# GENERAL COMMENTS #############################
# Dl._vdist_3min[simulation number, timestamp, link id]
# DL.segment_lengths[link id]


############################# BASIC PLOT FUNCTIONS #############################
def sublink(row):
    """
    Take row of links as such :
    
    for _, row in links.iterrows():
    
    gives back the correct x, y arrays
    """
    if row["out_ang"] > np.pi/2 or (row["out_ang"] < 0 and row["out_ang"]>= -np.pi/2):
        x = np.array([row["to_x"], row["from_x"]])
        y = np.array([row["from_y"], row["to_y"]])
    else:   
        x = np.array([row["from_x"], row["to_x"]])
        y = np.array([row["from_y"], row["to_y"]])
    return x,y

def link(ax, grad = False, color = "red", zorder = 2, norm=None, p=None, t=None, cmap = None):
    """
    THIS FUNCTION IS 
    ax : The current figure
    grad : bool
    color : Plot color
    zorder : The layer on which the links are plot
    
    The folowing are considered only if grad == True
    norm : The color scale, see mcolors.Normalize()
    p : The current parameter
    t = The current time frame
    cmap : The color map, see plt.get_cmap(")
    
    """   
    if not grad:
        for i, row in links.iterrows():
            x, y = sublink(row)
            ax.plot(x, y, c=color, linewidth=1, zorder = zorder) 
        return
            
    if norm is None or p is None or t is None:
        raise ValueError("norm, p and t are not given.")  
          
    elif grad :
        for j, row in links.iterrows():
            x = np.array([row["from_x"], row["to_x"]])
            y = np.array([row["from_y"], row["to_y"]])
            if pd.isna(p[0,t,j]):  
                z = "lime"
            else:
                if cmap is None:
                    raise ValueError("Colormap is missing")
                else:
                    z = cmap(norm(p[0,t,j]))
            ax.plot(x, y, c=z, linewidth=1)
        return


def polyg(ax, color = "blue", alpha= 1, zorder=3):
    """
    ax : The current figure
    color : Plot color
    zorder : The layer on which the links are plot
    """
    for section_id, data in DL.intersection_polygon.items():
        poly = data["polygon"]
        x = np.array([p[0] for p in poly] + [poly[0][0]])
        y = np.array([p[1] for p in poly] + [poly[0][1]])
        ax.plot(x, y, c = color, alpha= alpha, zorder = zorder)
    return


def numerical_sort(file):
    """
    This function sorts files in the correct numerical order
    
    Transforms
    [graph1.png, graph10.png, graph11.png, ..., graph19.png, graph2.png, graph20.png, ...]
    
    Into
    [graph1.png, graph2.png, ...]
    """
    
    numbers = re.findall(r'\d+', file) # We take the numbers of the file and put them in a list. Ex: graph6_time12.png --> [6, 12]
    if numbers:
        return int(numbers[0]) # If a number has been found, we take the first one and return it to be used as index to ensure correct order
    else: 
        return -1 # If no numbers has been found, return the index to be at last place


############################# CREATING THE BASELINE MAP #############################
def graph():    
    ### Creating the figure
    fig, ax = plt.subplots(dpi = 250)

    ### Plotting nodes
    ax.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=10, c = "black", alpha=1, zorder = 1)

    ### Plotting the links
    link(ax)        
        
    ### Plotting the intersetion polygons
    polyg(ax)
        
    ### Axis, labels, ration, sizes, ...
    ax.set_aspect("equal")
    ax.set_title("Nodes + Links + Intersection Polygons", fontsize=10)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    fig.savefig("graph.png")
    plt.close(fig)



############################# GRADIENT MAP #############################
def gradient_gif(param: list, param_name:list, fps : int):
    """
    Creates a gif made of the graphs of the gradient maps for each parameter
    param : list of parameters 
    param_name : list of parameters name, will impact folder names and plot titles
    fps : amount of images per second (higher fps = shorter gif)
    """
    
    i = -1 
    
    for pn in param_name:
        # We create a folder for each parameter to store our png
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", str(pn)), exist_ok=True)    
      
    for p in param:
        i += 1
        
        ### Subplot
        fig, ax = plt.subplots(dpi=150)
        
        ### Plotting nodes
        ax.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=10, c = "black", alpha=0.5, zorder = -2)
        
        ### Axis, labels, ratio, sizes, ...
        ax.set_aspect("equal")
        ax.set_title(f"{param_name[i]}")
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        
        ### Normalized color gradient based on the min and max values of p
        norm = mcolors.Normalize(
            np.nanmin(p[0,:]),
            np.nanmax(p[0,:])
            ) 
        
        ### Gif folder
        gif_folder = os.path.join(localgif)
        os.makedirs(gif_folder, exist_ok=True)
        gif_path = os.path.join(gif_folder, f"{param_name[i]}.gif")

        ### Plotting the intersetion polygons
        polyg(ax, "black", 1, 1)
        
        ### Chosing the colors
        cmap = cm.get_cmap("coolwarm")
        
        ### Storing the links coordinates              
        link_collections = []
        for j, row in links.iterrows():
            x, y = sublink(row)
            if pd.isna(p[0,0,j]):
                color = "lime"
            else:
                color = cmap(norm(p[0,0,j]))
            coll = ax.plot(x, y, c=color, linewidth=1)[0]  
            link_collections.append(coll)

        ### Colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=str(param_name[i]), location="right")

        ### Creating each frames
        with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
            for t in range(p.shape[1]):
                
                ### For every time measurement we rewrite the title to update the t value
                ax.set_title(f"{param_name[i]} @ {fps} fps : t = {t}", fontsize=10)
                
                ### Updating the colors
                for j, coll in enumerate(link_collections):
                    if pd.isna(p[0,t,j]):
                        color = "lime"
                    else:
                        color = cmap(norm(p[0,t,j]))
                    coll.set_color(color)
                    
                ### Actualize the canvas for each frame
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8) # Convert RGBA to RGB
                image = image.reshape(h, w, 4)[:, :, :3] # Reshaping the image
                
                # Creating a folder to store each frames
                os.makedirs(f"figure/{param_name[i]}", exist_ok = True)
                fig.savefig(f"figure/{param_name[i]}/graph{t}.png")
                
                writer.append_data(image) # Add the frame to the gif
                print(f"{param_name[i]} t={t}")

        plt.close(fig)
    return



# "vdist_3min_over_segment_lengths" is the amount of travels per link. Ex: 175 means that vehicles have travelled 175 times the link length
# "vtime_3min_over_3min" is the amount of vehicle on average on the link during the 3 mins. Ex: 40 means that there are on average 40 vehicles on the link

fps = 0.5

############################# KMEANS #############################
os.makedirs(f"figure/clustering", exist_ok = True)


############################# KMEANS - WITH VELOCITY / TRAFFIC FEATURES #############################

def kmeans_clustering(n_clusters, name, feature_type, spatial_weight=2.5, dynamic_weight = 1, random_state=42, threshold=np.array([]), filter=False, show=True, timeframe=None, init_links = None):
    """
    KMeans clustering using the same rich traffic feature vectors as the Ward
    method, making it directly comparable with `clustering()`.

    Unlike Ward, KMeans does NOT enforce network connectivity, so clusters may
    be spatially scattered but will group links with similar traffic behaviour.

    Parameters
    ----------
    n_clusters   : int – number of clusters
    name         : str – subfolder / filename prefix for saved figures
    feature_type : str – one of {"geometric", "speed", "distance", "time"}
    random_state : int – random seed for reproducibility (default 42)
    threshold    : array - links index with outside the threshold
    init_links   : list - links indices to 
    ─────────────────────────────────────────────────────────────────────────────
    HOW KMEANS WORKS — THE FOUR STEPS
    ─────────────────────────────────────────────────────────────────────────────
    KMeans is an iterative algorithm that partitions N data points into k groups
    by minimising the total within-cluster variance (inertia).  Each iteration
    consists of four conceptual steps:

      STEP 1 – INITIALISE CENTROIDS
          Place k centroids in the feature space.  By default sklearn uses
          "k-means++" which spreads the initial centroids far apart, dramatically
          reducing the chance of a bad local minimum compared to random placement.
          `n_init="auto"` tells sklearn to run the full algorithm several times
          with different initialisations and keep the best result.

      STEP 2 – LABEL EACH DATA POINT
          Assign every data point (here: every road link) to the nearest centroid
          using Euclidean distance in the feature space.  The feature space here
          encodes speed/distance/time temporal profiles, NOT raw geographic
          coordinates, so "nearest" means "most similar traffic behaviour".

      STEP 3 – UPDATE CENTROIDS
          Recompute each centroid as the mean of all data points currently
          assigned to it.  A centroid is just a vector of the same dimensionality
          as the feature matrix X, representing the "average traffic profile" of
          its cluster.

      STEP 4 – REPEAT UNTIL CONVERGENCE
          Steps 2 and 3 are repeated until centroid positions stop moving
          (change is below a tolerance) or a maximum number of iterations is
          reached.  sklearn's default tolerance is 1e-4 and default max_iter is
          300.  All of this is handled internally by sklearn.

    Because all four steps run inside `.fit_predict()`, the comments below
    point to the exact lines that trigger or prepare each step.
    ─────────────────────────────────────────────────────────────────────────────
    """

    # ── Output folder ──────────────────────────────────────────────────────────
    folder = f"figure/clustering/{name}"
    os.makedirs(folder, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 (PREPARATION) – BUILD THE FEATURE MATRIX
    # ══════════════════════════════════════════════════════════════════════════
    # `build_cluster_features` constructs the matrix X of shape (n_links, n_features).
    # Each row is one road link; the columns encode its traffic behaviour:
    #
    #   • For feature_type="speed":
    #       – mean speed across all time steps (how fast is this link on average?)
    #       – std of speed       (how much does speed vary throughout the day?)
    #       – peak timing        (when does congestion hit its worst point?)
    #       – 3 SVD components   (dominant temporal patterns, e.g. morning peak,
    #                             evening peak, flat profile)
    #       – x, y coordinates   (scaled by spatial_weight=1.2 to keep clusters
    #                             geographically coherent without being purely spatial)
    #
    # This is the raw material that KMeans will operate on.  All columns are
    # z-score normalised (StandardScaler) so that no single feature dominates
    # simply because of its units or magnitude.
    #
    # NOTE: even though we ask for feature_type="speed" here, the function also
    # accepts "distance" and "time", which swap in different traffic signals
    # while keeping the same temporal feature structure.
    
    one_tf = timeframe is not None
    X = build_cluster_features(feature_type, spatial_weight=2.5, dynamic_weight = 1, timeframe=timeframe if one_tf else 0, threshold=threshold, filter=filter, one_timeframe=one_tf)

    if feature_type == "speed" and threshold.size > 0:
            print(f"The low speed and short links are :{threshold}")
            
    # X.shape → (n_links, n_features)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 – INITIALISE CENTROIDS
    # ══════════════════════════════════════════════════════════════════════════
    # `KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")`
    # constructs the KMeans object but does NOT run the algorithm yet.
    #
    #   n_clusters=n_clusters
    #       How many centroids (k) to place.  Each centroid is a vector in the
    #       same feature space as X, representing the "centre of mass" of one
    #       cluster.
    #
    #   random_state=random_state
    #       Seeds the random number generator so the initialisation is
    #       reproducible.  Two runs with the same seed always produce the same
    #       centroids and therefore the same final clustering.
    #
    #   n_init="auto"
    #       sklearn will run the entire algorithm (steps 1–4) multiple times,
    #       each time starting from a different random initialisation, and will
    #       return the result with the lowest inertia (tightest clusters).
    #       This guards against converging to a poor local minimum.
    #
    # Initialisation strategy (k-means++, sklearn default):
    #       1. Pick the first centroid uniformly at random from the data points.
    #       2. For each remaining centroid, pick a data point with probability
    #          proportional to its squared distance from the nearest already-chosen
    #          centroid.  This spreads the initial centroids across the space,
    #          making convergence faster and results better on average.
    #
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 – LABEL EACH DATA POINT  (first pass)
    # STEP 3 – UPDATE CENTROIDS
    # STEP 4 – REPEAT UNTIL CONVERGENCE
    # ══════════════════════════════════════════════════════════════════════════
    # `.fit_predict(X)` triggers all remaining steps in one call:
    #
    #   fit():
    #     Iterates steps 2 and 3 until convergence:
    #       • E-step (Step 2): assign each link (row of X) to its nearest centroid
    #                          using Euclidean distance.
    #       • M-step (Step 3): recompute each centroid as the mean of all links
    #                          currently assigned to it.
    #     Stops when max(centroid shift) < tol (default 1e-4) OR max_iter (300)
    #     is reached.
    #
    #   predict():
    #     Returns the final cluster label (0 … k-1) for every data point.
    #     This is the result of the last E-step after convergence.
    #
    # `labels` is therefore a 1-D integer array of length n_links where
    # labels[i] is the cluster index (0 … n_clusters-1) of the i-th road link.
    spawn_links = []
    
    if init_links is not None:
        if len(init_links) != int(n_clusters):
            print(f"n_clusters doesn't match the amount of init_links")
            n_clusters = len(init_links)
            print(f"n_clusters set to {len(init_links)}")
        init_centroids = X[init_links]  
        n_clusters = len(init_links)
        kmeans = KMeans(
            n_clusters=n_clusters,      # ← STEP 1: sets the number of centroids
            init=init_centroids,        # ← STEP 1: seeds the initialisation
            n_init=1,                   # ← STEP 4: how many independent restarts
            random_state=random_state,  # ← STEPS 1–4: runs the full algorithm
        )
        
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
    )
        
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
                                
    if init_links is None:
        for k in range(n_clusters):
            cluster_idx = np.where(labels == k)[0]
            cluster_points = X[cluster_idx]
            centroid = centroids[k]

            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_idx = cluster_idx[np.argmin(distances)]

            spawn_links.append(closest_idx)
                                   
                                
                                
    
    # ── Attach cluster labels to the links DataFrame ───────────────────────────
    # Each road link now carries an integer cluster ID (0 … n_clusters-1).
    # This is the "labelling" result of Step 2 at convergence.
    plot_links = links.copy()
    
    ### FILTER ###
    if filter:
        if threshold.size > 0:
            mask = np.ones(plot_links.shape[0], dtype=bool)
            mask[threshold] = False 
            plot_links = plot_links.loc[mask] 
    ##############
    
    plot_links["cluster"] = labels  # ← STEP 2 result: final label for each link

    # ── Diagnostic: print the range of cluster sizes ───────────────────────────
    # np.bincount counts how many links belong to each cluster.
    # A very uneven distribution (e.g. one cluster with 2 links, another with 500)
    # is a signal that k may be too high or that the feature space has outliers.
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    print(f"{name}: KMeans clusters, size range {cluster_sizes.min()}-{cluster_sizes.max()}")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALISATION — map the cluster labels back onto the road network
    # ══════════════════════════════════════════════════════════════════════════

    # Build a discrete colour palette: one distinct colour per cluster.
    # tab10 has 10 maximally distinct colours; `.resampled(n_clusters)` picks
    # exactly n_clusters of them at equal spacing.
    cmap_discrete = matplotlib.colormaps.get_cmap("tab10").resampled(n_clusters)
    cluster_colors = cmap_discrete(np.linspace(0, 1, n_clusters))
    # cluster_colors[k] → RGBA tuple for cluster k

    # ── Figure setup ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(dpi=250)
    ax.set_aspect("equal")
    ax.set_title(f"{name} (KMeans, k={n_clusters})", fontsize=9)
    
    ### FILTER ###
    if filter:
        ax.set_title(f"{name}_filtered (KMeans, k={n_clusters})", fontsize=9)
    ##############    
    
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    # ── Draw each road link coloured by its cluster assignment ─────────────────
    # This loop translates the abstract cluster labels (integers) into a spatial
    # picture: links that share a similar speed/distance/time profile receive the
    # same colour regardless of where they sit in the network.
    # Line width encodes the number of lanes, providing an extra visual cue about
    # road hierarchy within each cluster.
    
    # PLOT LINKS
    for idx, row in plot_links.iterrows():
        x, y = sublink(row)

        color = cluster_colors[int(row["cluster"])]
        z = 1

        if idx in threshold:
            color = "black"
            z = 3

        ax.plot(x, y, c=color, linewidth=0.4 + row["num_lanes"] * 0.4, zorder=z)


    # PLOT SPAWN LINKS
    all_spawn = []
    
    if len(spawn_links) != 0:
        all_spawn.extend(spawn_links)

    if init_links is not None:
        all_spawn.extend(init_links)

    for idx in set(all_spawn): 
        row = plot_links.iloc[idx]
        x, y = sublink(row)

        ax.plot(
            x, y,
            c="lime",
            linewidth=2 + row["num_lanes"] * 0.4,
            zorder=4
        )


    # Filter (si activé) 
    if filter:
        for idx, row in links.loc[threshold].iterrows():
            x, y = sublink(row)
            ax.plot(
                x, y,
                c="black",
                linewidth=0.4 + row["num_lanes"] * 0.4,
                zorder=3
            )
    ##############
    
    # ── Overlay intersection polygons for spatial context ──────────────────────
    polyg(ax, color="black", alpha=0.6, zorder=-1)

    # ── Legend ─────────────────────────────────────────────────────────────────
    handles = []
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)
    speed = np.divide(
            vdist,
            vtime,
            out=np.full(vdist.shape, np.nan, dtype=float),
            where=vtime != 0,
        )
    speed = mean_over_sessions(np.nan_to_num(speed, nan=0.0))
    print(speed.shape)
    speed = np.mean(speed, axis = 0)
    print(speed.shape)
    
    for k in range(n_clusters):
        cluster_idx = np.where(labels == k)[0]          # indices des liens du cluster k
        mean_speed_real = speed[cluster_idx].mean()
        handles.append(
            plt.Line2D(
                [0], [0], 
                color=cluster_colors[k], 
                lw=3, 
                label=f"Cluster {k} – {mean_speed_real:.2f} m/s"
            )
        )
    handles.append(
    plt.Line2D([0], [0], color="lime", lw=3, label="Spawn points")
)
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    # ── Save ───────────────────────────────────────────────────────────────────
    if filter == True:
        if show:
            fig.savefig(f"{folder}/{n_clusters}_{name}_filtered.png")
            plt.close(fig)
            print(f"Saved → {folder}/{n_clusters}_{name}_filtered.png")
        else:
            fig.savefig(f"{folder}/{name}_filtered.png")
            plt.close(fig)
            print(f"Saved → {folder}/{name}_filtered.png")
            
    elif init_links is not None:
        if show:
            fig.savefig(f"{folder}/{n_clusters}_{name}_{str(init_links)}.png")
            plt.close(fig)
            print(f"Saved → {folder}/{n_clusters}_{name}_{str(init_links)}.png")
        else:
            fig.savefig(f"{folder}/{name}_{str(init_links)}.png")
            plt.close(fig)
            print(f"Saved → {folder}/{name}_{str(init_links)}.png")  
    
    else:
        if show:
            fig.savefig(f"{folder}/{n_clusters}_{name}_best.png")
            plt.close(fig)
            print(f"Saved → {folder}/{n_clusters}_{name}_best.png")
        else:
            fig.savefig(f"{folder}/{name}_best.png")
            plt.close(fig)
            print(f"Saved → {folder}/{name}_best.png")
    
    
            

############################# CLUSTERING (using a library) (WARD / AGGLOMERATIVE) #############################
os.makedirs(f"figure/clustering", exist_ok = True)

NETWORK_CONNECTIVITY = sparse.csr_matrix(DL.adjacency)


def mean_over_sessions(values):
    """
    Returns the mean of a value over all of the sessions.
    """
    valid_counts = np.sum(~np.isnan(values), axis=0)
    summed = np.nansum(values, axis=0)
    return np.divide(
        summed,
        valid_counts,
        out=np.full(summed.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )


def fill_profile_nans(profile):
    """
    Fills the NaNs with the median of each column of the profile, i.e. the 2D NumPy array of the mean_over_session of a parameter.
    """
    filled = profile.copy() 
    time_medians = np.nanmedian(filled, axis=0) 
    time_medians = np.where(np.isnan(time_medians), 0.0, time_medians) 
    nan_rows, nan_cols = np.where(np.isnan(filled)) 
    filled[nan_rows, nan_cols] = time_medians[nan_cols] 
    return filled


def rowwise_zscore(profile):
    """
    Returns row-wise the z-score normalization of a profile.
    """
    mean = profile.mean(axis=1, keepdims=True)
    std = profile.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (profile - mean) / std


def profile_components(profile, n_components=3):

    """
    Use SVD to return the profile's principal components.
    """
    centered = profile - profile.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(n_components, centered.shape[0], centered.shape[1])
    return u[:, :n_comp] * s[:n_comp]


def temporal_cluster_features(profile, peak_mode, spatial_weight=2.5, dynamic_weight = 1, threshold=np.array([]), filter=False): #trial of 0 so that it doesnt take into account the geometry of the map
    """
    Compute combined temporal and spatial features for clustering nodes based on
    their temporal profiles.

    The function extracts statistical and structural features from each profile including 
    mean, 
    standard deviation, 
    peak timing, 
    and principal components of the normalized profile (n_components).
    
    These dynamic features are combined with spatial node
    coordinates to produce a feature matrix suitable for clustering.
        """
    filled = fill_profile_nans(profile)
    peak_idx = np.argmin(filled, axis=1) if peak_mode == "min" else np.argmax(filled, axis=1)
    peak_time = peak_idx / max(filled.shape[1] - 1, 1)
    
    dynamic = np.column_stack([
        filled.mean(axis=1),
        filled.std(axis=1),
        peak_time,
        profile_components(rowwise_zscore(filled), n_components=8),# IMPORTANT TO CHANGE AS IT ONLY TAKES INTO ACCOUNT THE THREE VALUESSSSSSSS
    ])
    
    
    spatial_features = np.column_stack([links["c_x"].to_numpy(), links["c_y"].to_numpy()])
    
    ### FILTER ###
    if filter:
        threshold = threshold.astype(int)
        mask = np.ones(spatial_features.shape[0], dtype=bool)
        mask[threshold] = False 
        spatial_features = spatial_features[mask,:]
    ##############
    
    scaler = StandardScaler()

    dynamic_scaled = scaler.fit_transform(dynamic) * dynamic_weight
    spatial_scaled = scaler.fit_transform(spatial_features) * spatial_weight

    return np.hstack([dynamic_scaled, spatial_scaled])


def build_cluster_features(feature_type, timeframe, spatial_weight=2.5, dynamic_weight = 1, threshold=np.array([]), filter=False, one_timeframe = False):
    """
    Build feature matrices for clustering based on different traffic descriptors.

    Depending on the selected feature type, the function constructs node-level
    feature representations from geometric attributes or temporal traffic
    profiles derived from distance and travel time measurements.
    
    feature_type : {"geometric", "speed", "distance", "time"}
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    # to_remove = small_and_slow()
    # n_links = vdist.shape[2]

    # mask = np.ones(n_links, dtype=bool)
    # mask[to_remove] = False
    
    
    if feature_type == "geometric":
        geometric = np.column_stack([
            DL.node_coordinates[:, 0],
            DL.node_coordinates[:, 1],
            links["length"].to_numpy(dtype=float),
            links["num_lanes"].to_numpy(dtype=float),
        ])
        return StandardScaler().fit_transform(geometric)

    if feature_type == "speed":
        speed = np.divide(
            vdist,
            vtime,
            out=np.full(vdist.shape, np.nan, dtype=float),
            where=vtime != 0,
        )

        speed_profile = mean_over_sessions(speed)

        ### FILTER ###
        if filter:
            if threshold.size > 0:
                mask = np.ones(speed_profile.shape[1], dtype=bool)
                mask[threshold] = False
                speed_profile = speed_profile[:, mask]
        ##############

        if one_timeframe:
            #  speed snapshot 
            speed_snapshot = np.nan_to_num(speed_profile[timeframe, :], nan=0.0)

            #  spatial features 
            spatial = np.column_stack([
                links["c_x"].to_numpy(),
                links["c_y"].to_numpy()
            ])

            #  optional filtering
            if filter and threshold.size > 0:
                mask = np.ones(len(speed_snapshot), dtype=bool)
                mask[threshold] = False

                speed_snapshot = speed_snapshot[mask]
                spatial = spatial[mask, :]

            #  build feature matrix 
            X = np.column_stack([speed_snapshot, spatial])

            #  normalize 
            X = StandardScaler().fit_transform(X)


            X[:, 0] *= dynamic_weight
            X[:, 1:] *= spatial_weight

            return X

        return temporal_cluster_features(speed_profile.T, peak_mode="min", threshold=threshold, filter=filter)

    if feature_type == "distance":
        distance = np.where(vdist != 0, vdist, np.nan)
        distance_profile = np.log1p(mean_over_sessions(distance)).T
        return temporal_cluster_features(distance_profile, peak_mode="max")

    if feature_type == "time":
        time_profile = np.log1p(mean_over_sessions(np.where(vtime != 0, vtime, np.nan))).T
        return temporal_cluster_features(time_profile, peak_mode="max")
        
    raise ValueError(f"Unknown feature_type: {feature_type}")

def clustering(n_clusters, name, feature_type, threshold=np.array([])):
    """
    Perform hierarchical (Ward) clustering of network links and visualize the result. 

    Uses AgglomerativeClustering with Ward linkage and a network connectivity
    constraint so that only spatially adjacent links can be merged.  This
    guarantees that every resulting cluster is a contiguous subgraph of the
    road network.

    Parameters
    ----------
    n_clusters   : int  – desired number of clusters
    name         : str  – subfolder / filename prefix for saved figures
    feature_type : str  – one of {"geometric", "speed", "distance", "time"}
    threshold    : array - links index with outside the threshold
    """
    n_clus = n_clusters
    
    ### Creating folder
    folder = f"figure/clustering/{name}"
    os.makedirs(folder, exist_ok=True)

    ### Clustering and outliers
    X = build_cluster_features(feature_type)
       
    if feature_type == "speed" and threshold.size > 0:   
        print(f"The low speed and short links are :{threshold}")

    labels = AgglomerativeClustering( # Assign a cluster to each link, euclidian distance
        n_clusters=n_clus,
        linkage="ward",
        connectivity=NETWORK_CONNECTIVITY,
    ).fit_predict(X)
    
    plot_links = links.copy()
    plot_links["cluster"] = labels
    cluster_sizes = np.bincount(labels, minlength=n_clus)
    print(f"{name}: connected Ward clusters, size range {cluster_sizes.min()}-{cluster_sizes.max()}")

    cmap_discrete = matplotlib.colormaps.get_cmap("tab10").resampled(n_clus)
    cluster_colors = cmap_discrete(np.linspace(0, 1, n_clus))
    fig, ax = plt.subplots(dpi=250)
    ax.set_aspect("equal")
    ax.set_title(f"{name} (connected Ward, k={n_clus})", fontsize=9)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis='both', labelsize=8)


    for idx, row in plot_links.iterrows():
        z = 1
        x, y = sublink(row)
        color = cluster_colors[int(row["cluster"])]
        if idx in threshold:
            color = "black"
            z = 3 
        ax.plot(x, y, c=color, linewidth = 0.4 + row["num_lanes"] * 0.4, zorder = z)


    handles = [
        plt.Line2D([0], [0], color=cluster_colors[k], lw=3, label=f"Cluster {k}")
        for k in range(n_clus)
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    fig.savefig(f"{folder}/{name}_best.png")
    plt.close(fig)
    print(f"Saved → {folder}/{name}_best.png")

    
### Small links grouping

def thresholds(max_speed = 0, max_length = 0):
    """
    Finds the links with an 85th-percentile mean over every sessions speed lower than max_speed and a length smaller than max_length

    Parameters
    ----------
    max_speed :  int - maximum 85th-percentile mean over every sessions speed [m/s] set to np.inf if no speed restrictions are wanted, i.e. only short links
    max_length : int - maximum link length [m], set to np.inf if no length restrictions are wanted, i.e. only slow links
    """
    # Dl._vdist_3min[simulation number, timestamp, link id]
    # DL.segment_lengths[link id]
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)
     
    speed = np.divide(
            vdist,
            vtime,
            out=np.full(vdist.shape, np.nan, dtype=float),
            where=vtime != 0,
        )
    
    speed =  mean_over_sessions(np.nan_to_num(speed, nan=0.0))
    speed_85 = np.percentile(speed, 85, axis=0)

    low_speed_links = np.where((speed_85 <= max_speed))[0]
    
    smallslow = [link for link in low_speed_links if links.iloc[link]["length"] <= max_length]
            
    return np.array(smallslow)


### Simplified map: merge parallel/duplicate segments and show average speed

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def group_segments(distance_threshold=35.0, lateral_threshold=15.0):
    """
    Groups road segments that represent the same road.

    Three matching criteria (any one is sufficient):
    1. Forward match: both endpoint pairs within distance_threshold
    2. Reverse match: endpoints cross-match (opposite direction)
    3. Lateral match: midpoint-to-midpoint distance within lateral_threshold
       (handles segments of different lengths on the same road)

    Parameters
    ----------
    distance_threshold : float - endpoint proximity threshold [m]
    lateral_threshold : float - max distance between midpoints [m]
    """
    N = len(links)
    from_xy = links[["from_x", "from_y"]].to_numpy()
    to_xy = links[["to_x", "to_y"]].to_numpy()

    from_tree = cKDTree(from_xy)
    to_tree = cKDTree(to_xy)

    # --- 1. Forward match: from_i ~ from_j AND to_i ~ to_j ---
    from_from_pairs = from_tree.query_pairs(distance_threshold)
    to_to_pairs = to_tree.query_pairs(distance_threshold)
    forward_matches = from_from_pairs & to_to_pairs

    # --- 2. Reverse match: from_i ~ to_j AND to_i ~ from_j ---
    from_to_neighbors = from_tree.query_ball_tree(to_tree, distance_threshold)
    to_from_neighbors = to_tree.query_ball_tree(from_tree, distance_threshold)

    reverse_matches = set()
    for i in range(N):
        candidates = set(from_to_neighbors[i]) & set(to_from_neighbors[i])
        for j in candidates:
            if i != j:
                reverse_matches.add((min(i, j), max(i, j)))

    # --- 3. Lateral match: midpoints within lateral_threshold ---
    mid_xy = (from_xy + to_xy) / 2.0
    center_tree = cKDTree(mid_xy)
    lateral_matches = center_tree.query_pairs(lateral_threshold)

    all_matches = forward_matches | reverse_matches | lateral_matches

    uf = UnionFind(N)
    for i, j in all_matches:
        uf.union(i, j)

    groups_dict = {}
    for i in range(N):
        root = uf.find(i)
        groups_dict.setdefault(root, []).append(i)

    return list(groups_dict.values())


def compute_group_speeds(groups):
    """
    Computes the global average speed (nanmean over all sessions, timesteps, and group members)
    for each group of segments.

    Returns an array of shape (len(groups),) with mean speed [m/s] per group.
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist,
        vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=vtime != 0,
    )

    group_speeds = np.empty(len(groups))
    for k, group in enumerate(groups):
        group_speeds[k] = np.nanmean(speed[:, :, group])

    return group_speeds


def simplified_map(distance_threshold, grad=True, color="navy"):
    """
    Creates a simplified road network map by merging parallel/duplicate segments.

    Parameters
    ----------
    distance_threshold : float - endpoint proximity threshold [m]
    grad : bool - if True, color by average speed; if False, use flat color
    color : str - flat color when grad=False
    """
    groups = group_segments(distance_threshold)
    representatives = []
    for group in groups:
        best = max(group, key=lambda idx: links.iloc[idx]["num_lanes"])
        representatives.append(best)

    fig, ax = plt.subplots(dpi=250)

    if grad:
        speeds = compute_group_speeds(groups)
        valid_speeds = speeds[~np.isnan(speeds)]
        norm = mcolors.Normalize(vmin=np.nanmin(valid_speeds), vmax=np.nanmax(valid_speeds))
        cmap = plt.get_cmap("RdYlGn")

        for k, rep_idx in enumerate(representatives):
            row = links.iloc[rep_idx]
            x, y = sublink(row)
            if np.isnan(speeds[k]):
                c = "lime"
            else:
                c = cmap(norm(speeds[k]))
            ax.plot(x, y, c=c, linewidth=0.5)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Average speed [m/s]")
        suffix = "speed"
    else:
        for rep_idx in representatives:
            row = links.iloc[rep_idx]
            x, y = sublink(row)
            ax.plot(x, y, c=color, linewidth=0.3)
        suffix = "flat"

    polyg(ax, color="black", alpha=0.3, zorder=-1)

    ax.set_aspect("equal")
    ax.set_title(f"Simplified network (threshold={distance_threshold}m, {len(representatives)}/{len(links)} segments)", fontsize=9)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)

    fig.savefig(f"{localfigure}/simplified_map_{suffix}.png")
    plt.close(fig)
    print(f"Saved simplified map ({suffix}): {len(groups)} groups from {len(links)} segments")


############################# PERCOLATION ANALYSIS (Li et al. 2015) #############################

def build_directed_adjacency():
    """Builds a directed adjacency matrix from the connections DataFrame."""
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
    Computes normalized speed r_ij(t) = v_ij(t) / v_max_ij for each segment.
    v_max_ij is the 95th percentile speed of segment j (across all sessions and timesteps).

    Returns r of shape (S, T, N) with values in [0, 1].
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist, vtime,
        out=np.full(vdist.shape, np.nan, dtype=float),
        where=vtime != 0,
    )

    # 95th percentile per segment (across all sessions and timesteps)
    v_max = np.nanpercentile(speed.reshape(-1, speed.shape[2]), 95, axis=0)
    v_max[v_max == 0] = np.nan

    r = speed / v_max[np.newaxis, np.newaxis, :]
    r = np.clip(r, 0, 1)
    return r


def percolation_sweep(r_t, adj_directed, q_values):
    """
    For a single time snapshot r_t (shape N,), sweep over q values
    and find strongly connected components.

    Returns:
        giant : array of giant component sizes for each q
        second : array of second-largest component sizes for each q
    """
    N = r_t.shape[0]
    giant = np.zeros(len(q_values))
    second = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        functional = r_t >= q
        # Subgraph: keep only edges between functional nodes
        func_idx = np.where(functional)[0]
        if len(func_idx) == 0:
            continue

        sub_adj = adj_directed[np.ix_(func_idx, func_idx)]
        n_comp, labels = connected_components(sub_adj, directed=True, connection='strong')

        comp_sizes = np.bincount(labels)
        sorted_sizes = np.sort(comp_sizes)[::-1]
        giant[i] = sorted_sizes[0]
        if len(sorted_sizes) > 1:
            second[i] = sorted_sizes[1]

    return giant, second


def find_critical_threshold(r_t, adj_directed, q_values):
    """
    Finds q_c: the threshold where the second-largest strongly connected
    component is maximal (percolation transition point).

    Returns q_c, giant sizes array, second-largest sizes array.
    """
    giant, second = percolation_sweep(r_t, adj_directed, q_values)
    qc_idx = np.argmax(second)
    return q_values[qc_idx], giant, second


def find_bottlenecks(r_t, adj_directed, qc, delta=0.01):
    """
    Identifies bottleneck links by comparing the functional network just below
    and just above q_c. Bottlenecks are segments that are functional at (qc - delta)
    but dysfunctional at (qc + delta).

    Returns array of bottleneck segment indices.
    """
    functional_below = r_t >= (qc - delta)
    functional_above = r_t >= (qc + delta)

    # Links present below qc but removed above qc
    bottlenecks = np.where(functional_below & ~functional_above)[0]
    return bottlenecks


def percolation_analysis(session=0, timestep=None, n_q=100):
    """
    Full percolation analysis pipeline for a given session and timestep.

    Parameters
    ----------
    session : int - simulation session index
    timestep : int or None - if None, averages r over all timesteps
    n_q : int - number of q values to sweep
    """
    r = compute_normalized_speed()
    adj_directed = build_directed_adjacency()
    q_values = np.linspace(0, 1, n_q)

    if timestep is not None:
        r_t = r[session, timestep, :]
        # Replace NaN with 0 (no data = dysfunctional)
        r_t = np.nan_to_num(r_t, nan=0.0)
        time_label = f"session {session}, t={timestep}"
    else:
        # Average over all sessions and timesteps
        r_t = np.nanmean(r.reshape(-1, r.shape[2]), axis=0)
        r_t = np.nan_to_num(r_t, nan=0.0)
        time_label = "mean over all sessions/timesteps"

    # Find q_c
    qc, giant, second = find_critical_threshold(r_t, adj_directed, q_values)
    print(f"Critical threshold q_c = {qc:.3f} ({time_label})")

    # Find bottlenecks
    bottlenecks = find_bottlenecks(r_t, adj_directed, qc)
    print(f"Found {len(bottlenecks)} bottleneck segments at q_c")

    # --- Plot 1: Giant and second-largest component vs q ---
    fig, ax = plt.subplots(dpi=250)
    ax.plot(q_values, giant, label="Giant component (G)", color="blue")
    ax.plot(q_values, second, label="2nd largest (SG)", color="red")
    ax.axvline(qc, color="gray", linestyle="--", linewidth=0.8, label=f"$q_c$ = {qc:.3f}")
    ax.set_xlabel("Threshold q", fontsize=10)
    ax.set_ylabel("Component size (# segments)", fontsize=10)
    ax.set_title("Percolation transition", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    fig.savefig(f"{localfigure}/percolation_transition.png")
    plt.close(fig)
    print(f"Saved {localfigure}/percolation_transition.png")

    # --- Plot 2: Network at q_c with bottlenecks highlighted ---
    fig, ax = plt.subplots(dpi=250)

    functional = r_t >= qc

    # Plot all segments light gray
    for i, row in links.iterrows():
        x, y = sublink(row)
        ax.plot(x, y, c="lightgray", linewidth=0.3, zorder=1)

    # Plot functional segments colored by cluster
    func_idx = np.where(functional)[0]
    if len(func_idx) > 0:
        sub_adj = adj_directed[np.ix_(func_idx, func_idx)]
        n_comp, labels = connected_components(sub_adj, directed=True, connection='strong')
        comp_sizes = np.bincount(labels)
        # Sort clusters by size (descending) and assign colors to top clusters
        size_order = np.argsort(comp_sizes)[::-1]
        cluster_colors = ["green", "blue", "orange", "purple", "cyan"]

        for k, cluster_id in enumerate(size_order[:5]):
            members = func_idx[labels == cluster_id]
            c = cluster_colors[k] if k < len(cluster_colors) else "gray"
            for idx in members:
                row = links.iloc[idx]
                x, y = sublink(row)
                ax.plot(x, y, c=c, linewidth=0.5, zorder=2)

    # Highlight bottlenecks in red
    for idx in bottlenecks:
        row = links.iloc[idx]
        x, y = sublink(row)
        ax.plot(x, y, c="red", linewidth=1.5, zorder=3)

    ax.set_aspect("equal")
    ax.set_title(f"Network at $q_c$={qc:.3f}, bottlenecks in red ({len(bottlenecks)})", fontsize=9)
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    fig.savefig(f"{localfigure}/percolation_bottlenecks.png")
    plt.close(fig)
    print(f"Saved {localfigure}/percolation_bottlenecks.png")

    return qc, bottlenecks


def congestion_map(qc, session=0, timesteps=[0, 9, 23, 31,33,36,38]):
    """
    Plots congestion maps at selected timesteps using the critical threshold qc
    from percolation analysis. Congested segments (r < qc) are shown in red,
    functional segments (r >= qc) in green.

    Parameters
    ----------
    qc : float - critical threshold from percolation_analysis
    session : int - simulation session index
    timesteps : list - timestep indices to plot
    """
    r = compute_normalized_speed()
    folder = f"{localfigure}/congestion_maps"
    os.makedirs(folder, exist_ok=True)

    for t in timesteps:
        r_t = np.nan_to_num(r[session, t, :], nan=0.0)

        fig, ax = plt.subplots(dpi=250)

        for i, row in links.iterrows():
            x, y = sublink(row)
            if r_t[i] < qc:
                ax.plot(x, y, c="red", linewidth=1, zorder=2)
            else:
                ax.plot(x, y, c="green", linewidth=1, zorder=1)

        polyg(ax, color="black", alpha=0.3, zorder=-1)

        n_congested = np.sum(r_t < qc)
        ax.set_aspect("equal")
        ax.set_title(f"Congestion at t={t} ({t*3}min) — {n_congested}/{len(links)} congested, $q_c$={qc:.3f}", fontsize=8)
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

        # Legend
        handles = [
            plt.Line2D([0], [0], color="red", lw=2, label=f"Congested (r < {qc:.2f})"),
            plt.Line2D([0], [0], color="green", lw=2, label=f"Functional (r ≥ {qc:.2f})"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")

        fig.savefig(f"{folder}/congestion_t{t}.png")
        plt.close(fig)
        print(f"Saved congestion map t={t} ({t*3}min): {n_congested} congested segments")


def grid_clust(xdiv = 4, ydiv = 4, percentile = 65):
    """
    Clusters the links based on a rectangular grid
    
    xdiv : int
    """
    links_local = links.copy()
    tol = 100
    x_min = np.min(links_local["from_x"]) - tol
    x_max = np.max(links_local["to_x"]) + tol
    y_min = np.min(links_local["from_y"]) - tol
    y_max = np.max(links_local["to_y"]) + tol
 
    w = (x_max - x_min)/xdiv
    h = (y_max - y_min)/ydiv
    xs = np.arange(x_min, x_max, w)
    ys = np.arange(y_min, y_max, h)
    
    folder = f"figure/clustering/grid_clusters"
    os.makedirs(folder, exist_ok=True)
    
    fig, ax = plt.subplots(dpi = 250)
    
    ### Assigning the grid cell in which link is (clustering)
    links_local["cell_x"] = ((links_local["c_x"] - x_min)//w).astype(int)
    links_local["cell_y"] = ((links_local["c_y"] - y_min)//h).astype(int)
    

    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    speed = np.divide(
        vdist,
        vtime,
        out=np.full(vdist.shape, np.nan),  # fallback
        where=(vtime != 0) & (~np.isnan(vtime))
    )
    
    speed = mean_over_sessions(np.nan_to_num(speed, nan=0.0)) # (T, N)
    print(speed.shape)
    
    average_speed_per_link = np.nanmean(speed, axis=0)  #  Average speed over 2 hours (N, )
    links_local["speed_mean"] = average_speed_per_link
    
    cell_speed = (
        links_local
        .groupby(["cell_x", "cell_y"])["speed_mean"]
        .mean()
        .reset_index(name="cell_avg_speed")
    )
    
    links_local = links_local.merge(
        cell_speed,
        on=["cell_x", "cell_y"],
        how="left"
    )

    perc = np.percentile(average_speed_per_link, percentile, axis=0) # NN-th percentile 
    print(perc)
    
    links_local["color"] = np.where(
        links_local["cell_avg_speed"] >= perc,
        "green",
        "red"
    )
    ### Plotting the links
    for _, row in links_local.iterrows():

        ax.plot(
            [row["from_x"], row["to_x"]],
            [row["from_y"], row["to_y"]],
            color= row["color"]
        )

    ### Plotting the grid cells
    for x in xs:
        for y in ys:
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                edgecolor="black",
                facecolor="none",
                linewidth=0.5
            )
            ax.add_patch(rect)
            
    ### Plotting the intersections
    polyg(ax, color="black", zorder=-2)
    
    ax.set_title(f"{percentile}th percentile : Speed = {perc:.2f} m/s")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    fig.savefig(f"{folder}/grid{percentile}.png")
    plt.close()


def closest_link(x, y):
    """
    Returns the id of the closest link to the two coordinates given.
    Uses L2 norm AKA Euclidean distance
    
    Parameters
    ----------
    x : int - x coordinates
    y : int - y coordinates
    """

    coords = np.column_stack([links["c_x"].to_numpy(), links["c_y"].to_numpy()])
    
    point = np.array([x, y])
    
    distances = np.linalg.norm(coords - point, axis=1)
    
    return np.argmin(distances)



    


#graph()
param = [
    DL._vdist_3min, 
    DL._vtime_3min,
    DL._vdist_3min/DL._vtime_3min
    ]

param_name = [
    "vdist_3min", 
    "vtime_3min",
    "vdist_3min_over_vtime_3min"
    ]
#gradient_gif(param, param_name, fps)



n_clus = 8
seeds = np.linspace(0, 9, 10)
# grid_clust(20, 16, 85)
# grid_clust(20, 16, 65)
# grid_clust(20, 16, 55)
# grid_clust(20, 16, 45)

### Threshold
th = thresholds(max_speed=2, max_length=np.inf)
#print(th)

# ### Ward (agglomerative) – feature-driven, connectivity-constrained
# # clustering(n_clus, "geometric_clusters", "geometric")
# # clustering(n_clus, "distance_clusters", "distance")
# # clustering(n_clus, "time_clusters", "time")
# # clustering(n_clus, "speed_clusters", "speed", threshold=th)

# ## KMeans with velocity/traffic features  ← NEW
# kmeans_clustering(n_clus, "kmeans_speed_clusters",    "speed", threshold=th)
# kmeans_clustering(n_clus, "kmeans_speed_clusters",    "speed", threshold=th, filter=True)
# # kmeans_clustering(n_clus, "kmeans_distance_clusters", "distance")
# # kmeans_clustering(n_clus, "kmeans_time_clusters",     "time")

# ### Simplified map
# simplified_map(distance_threshold=50.0, grad=True)
# simplified_map(distance_threshold=50.0, grad=False, color="navy")


### Percolation analysis
#qc, bottlenecks = percolation_analysis(session=0)
#congestion_map(qc, session=0)


##################### CLUSTERS SPAWN POINTS/AMOUNT #####################


x_min_lim = np.min(links["from_x"])
x_max_lim = np.max(links["to_x"])
y_min_lim = np.min(links["from_y"])
y_max_lim = np.max(links["to_y"])

x25 = x_min_lim + 0.25 * (x_max_lim - x_min_lim)
x50 = x_min_lim + 0.5 * (x_max_lim - x_min_lim)
x75 = x_min_lim + 0.75 * (x_max_lim - x_min_lim)


y25 = y_min_lim + 0.25 * (y_max_lim - y_min_lim)
y50 = y_min_lim + 0.5 * (y_max_lim - y_min_lim)
y75 = y_min_lim + 0.75 * (y_max_lim - y_min_lim)

xs1 = [x25, x75]
ys1 = [y25, y75]

xs2 = [x50]
ys2 = [y25, y75]

xs3 = [x25, x75]
ys3 = [y50]

links_spawn_1 = [(x, y) for x in xs1 for y in ys1]
links_spawn_2 = [(x, y) for x in xs2 for y in ys2]
links_spawn_3 = [(x, y) for x in xs3 for y in ys3]

init_links_1 = [closest_link(x, y) for x, y in links_spawn_1]
init_links_2 = [closest_link(x, y) for x, y in links_spawn_2]
init_links_3 = [closest_link(x, y) for x, y in links_spawn_3]


cluster_amount = list(range(2,8,1))


for i in cluster_amount:
    kmeans_clustering(i, "kmeans_speed_clusters", "speed")
    kmeans_clustering(i, "kmeans_speed_clusters_t_10", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=10)
    kmeans_clustering(i, "kmeans_speed_clusters_t_30", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=30)

kmeans_clustering(5, "kmeans_speed_clusters_t_10_set_spawn", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=10, init_links=init_links_1)
kmeans_clustering(len(init_links_2), "kmeans_speed_clusters_t_10_set_spawn", "speed",  dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=10, init_links=init_links_2)
kmeans_clustering(len(init_links_3), "kmeans_speed_clusters_t_10_set_spawn", "speed",  dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=10, init_links=init_links_3)


kmeans_clustering(len(init_links_1), "kmeans_speed_clusters_t_30_set_spawn", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=30, init_links=init_links_1)
kmeans_clustering(len(init_links_2), "kmeans_speed_clusters_t_30_set_spawn", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=30, init_links=init_links_2)
kmeans_clustering(len(init_links_3), "kmeans_speed_clusters_t_30_set_spawn", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=30, init_links=init_links_3)




main_roads1 = [
    closest_link(429250, 4581750),
    closest_link(432200,4581900),
    closest_link(430280, 4582250) 
    ]

kmeans_clustering(len(main_roads1), "kmeans_speed_clusters_t_30_set_spawn", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=30, init_links=main_roads1)

main_roads2 = [
    closest_link(429250, 4581750),
    closest_link(431250, 4582350)
]

kmeans_clustering(len(main_roads2), "kmeans_speed_clusters_t_30_set_spawn", "speed", dynamic_weight = 1.5, spatial_weight = 2.0, timeframe=30, init_links=main_roads2)



#  WEIGHTS 

# UNIQUE TIMEFRAME
# dynamic_weight = 1.0
# spatial_weight = 2.5

# UNIQUE TIMEFRAME
# dynamic_weight = 1.5
# spatial_weight = 2.0