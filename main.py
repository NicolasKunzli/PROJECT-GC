import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
from DataLoad import DataLoader
import imageio
import re


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
links = pd.read_csv(os.path.join(path, "metadata", "link_bboxes.csv"))
connections = pd.read_csv(os.path.join(path, "metadata", "connections.csv"))

polygons = pd.read_json(os.path.join(path, "metadata", "intersec_polygon.json"))
lane_info = pd.read_json(os.path.join(path, "metadata", "lane_info.json"))
od = pd.read_json(os.path.join(path, "metadata", "od_pairs.json"))

with open (os.path.join(path, "metadata", "train_test_split.json")) as f:tts=f.read()
with open(os.path.join(path, "metadata", "sections_of_interest.txt"), "r") as f:interest = [line.strip() for line in f.readlines()]

### Class
DL = DataLoader()
DL.init_graph_structure


############################# GENERAL COMMENTS #############################
# Dl._vdist_3min[simulation number, timestamp, link id]
# DL.segment_lengths[link id]


############################# BASIC PLOT FUNCTIONS #############################
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

            x = np.array([row["from_x"], row["to_x"]])
            y = np.array([row["from_y"], row["to_y"]])   
 
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
            x = np.array([row["from_x"], row["to_x"]])
            y = np.array([row["from_y"], row["to_y"]])
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
                writer.append_data(image) # Add the frame to the gif
                print(f"{param_name[i]} t={t}")

        plt.close(fig)
    return

param = [
    DL._vdist_3min/DL.segment_lengths, 
    DL._vtime_3min/180,
    DL._vdist_3min, 
    DL._vtime_3min
    ]

param_name = [
    "vdist_3min_over_segment_lengths", 
    "vtime_3min_over_3min", 
    "vdist_3min", 
    "vtime_3min"
    ]

# "vdist_3min_over_segment_lengths" is the amount of travels per link. Ex: 175 means that vehicles have travelled 175 times the link length
# "vtime_3min_over_3min" is the amount of vehicle on average on the link during the 3 mins. Ex: 40 means that there are on average 40 vehicles on the link

fps = 0.5
gradient_gif(param, param_name, fps)