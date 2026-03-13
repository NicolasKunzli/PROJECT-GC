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
from sklearn.cluster import AgglomerativeClustering
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

# Replace NaN values by 0
'''DL._vdist_3min = np.nan_to_num(DL._vdist_3min, nan=0)
DL._vtime_3min = np.nan_to_num(DL._vtime_3min, nan=0)'''

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



############################# CLUSTERING #############################
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


def temporal_cluster_features(profile, peak_mode, spatial_weight=1.2):
    """
    Compute combined temporal and spatial features for clustering nodes based on
    their temporal profiles.

    The function extracts statistical and structural features from each profile including 
    mean, 
    standard deviation, 
    peak timing, 
    and principal components of the normalized profile. 
    
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
        profile_components(rowwise_zscore(filled), n_components=3),
    ])
    spatial_features = DL.node_coordinates.astype(float)
    return np.hstack([
        StandardScaler().fit_transform(dynamic),
        StandardScaler().fit_transform(spatial_features) * spatial_weight,
    ])


def build_cluster_features(feature_type):
    """
    Build feature matrices for clustering based on different traffic descriptors.

    Depending on the selected feature type, the function constructs node-level
    feature representations from geometric attributes or temporal traffic
    profiles derived from distance and travel time measurements.
    
    feature_type : {"geometric", "speed", "distance", "time"}
    """
    vdist = DL._vdist_3min.astype(float)
    vtime = DL._vtime_3min.astype(float)

    if feature_type == "geometric":
        geometric = np.column_stack([
            links["c_x"],
            links["c_y"],
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
        speed_profile = mean_over_sessions(speed).T
        return temporal_cluster_features(speed_profile, peak_mode="min")

    if feature_type == "distance":
        distance = np.where(vdist != 0, vdist, np.nan)
        distance_profile = np.log1p(mean_over_sessions(distance)).T
        return temporal_cluster_features(distance_profile, peak_mode="max")

    if feature_type == "time":
        time_profile = np.log1p(mean_over_sessions(np.where(vtime != 0, vtime, np.nan))).T
        return temporal_cluster_features(time_profile, peak_mode="max")

    raise ValueError(f"Unknown feature_type: {feature_type}")


def clustering(n_clusters, name, feature_type):
    """
    Perform hierarchical clustering of network links and visualize the result.
    """
    n_clus = n_clusters
    
    ### Creating folder
    folder = f"figure/clustering/{name}"
    os.makedirs(folder, exist_ok=True)

    ### Clustering
    X = build_cluster_features(feature_type)
    labels = AgglomerativeClustering( # Assign a cluster to each link
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

    for _, row in plot_links.iterrows():
        x, y = sublink(row)
        color = cluster_colors[int(row["cluster"])]
        ax.plot(x, y, c=color)

    handles = [
        plt.Line2D([0], [0], color=cluster_colors[k], lw=3, label=f"Cluster {k}")
        for k in range(n_clus)
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")

    fig.savefig(f"{folder}/{name}_best.png")
    plt.close(fig)
    print(f"Saved → {folder}/{name}_best.png")


def grid_clust(xdiv = 4, ydiv = 4, showgrid = True):
    """
    Clusters the links based on a rectangular grid
    
    xdiv : Number of cells on the x-axis
    xdiv : Number of cells on the y-axis
    """
    tol = 100
    x_min = np.min(links["from_x"]) - tol
    x_max = np.max(links["to_x"]) + tol
    y_min = np.min(links["from_y"]) - tol
    y_max = np.max(links["to_y"]) + tol
 
    w = (x_max - x_min)/xdiv # Width of a cell
    h = (y_max - y_min)/ydiv # Height of a cell
    
    xs = np.arange(x_min, x_max, w)
    ys = np.arange(y_min, y_max, h)
    
    ### Creating folder
    folder = f"figure/clustering/grid_clusters"
    os.makedirs(folder, exist_ok=True)
    
    fig, ax = plt.subplots(dpi = 250)
    
    ### Making a copy so that the clustering assignment isn't kept from one method to another
    plot_links = links.copy()

    ### Assigning the grid cell in which link is (clustering)
    plot_links["cell_x"] = ((links["c_x"] - x_min)//w) # Floor dividing by the width/height to assign a cell id on the x or y axis to each links
    plot_links["cell_y"] = ((links["c_y"] - y_min)//h)
    
    ### Manually assigning a color for each grid cell 
    cells = list(zip(plot_links["cell_x"], plot_links["cell_y"]))
    unique_cells = list(set(cells))

    cmap = plt.colormaps.get_cmap("tab20")
    cell_color = {cell: cmap(i) for i, cell in enumerate(unique_cells)}
    
    ### Plotting the links
    for _, row in plot_links.iterrows():

        cell = (row["cell_x"], row["cell_y"])
        color = cell_color[cell]

        ax.plot(
            [row["from_x"], row["to_x"]],
            [row["from_y"], row["to_y"]],
            color=color
        )
    
    ### Plotting the grid cells
    if showgrid == True:
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
    
    ax.set_title(f"Grid clustering of shape ({xdiv},{ydiv})")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    fig.savefig(f"{folder}/grid.png")
    plt.close()
    
    
n_clus = 8

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

grid_clust(4, 3)
clustering(n_clus, "geometric_clusters", "geometric")
clustering(n_clus, "distance_clusters", "distance")
clustering(n_clus, "time_clusters", "time")
clustering(n_clus, "speed_clusters", "speed")

