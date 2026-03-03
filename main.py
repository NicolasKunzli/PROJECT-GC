import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import sys
from DataLoad import DataLoader
import imageio
import re


### Loading files
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

### Instance of the class
DL = DataLoader()
DL.init_graph_structure





### Creating the figure
fig, ax = plt.subplots(dpi = 250)

### Plotting nodes
ax.scatter(DL.node_coordinates[:,0]/1000, DL.node_coordinates[:,1]/1000, s=10, c = "black", alpha=1, zorder = 1)

### Plotting the links
def link(ax, grad = False, color = "red", zorder = 2, norm=None, p=None, t=None, cmap = None):
    """
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
            x /= 1000
            y /= 1000     
            ax.plot(x, y, c=color, linewidth=1, zorder = zorder) 
        return
            
    if norm is None or p is None or t is None:
        raise ValueError("norm, p and t are not given. grad should be False")  
          
    elif grad :
        for j, row in links.iterrows():
            x = np.array([row["from_x"], row["to_x"]])
            y = np.array([row["from_y"], row["to_y"]])
            x /= 1000
            y /= 1000 
            if pd.isna(p[0,t,j]):  
                z = "lime"
            else:
                if cmap is None:
                    raise ValueError("Colormap is missing")
                else:
                    z = cmap(norm(p[0,t,j]))
            ax.plot(x, y, c=z, linewidth=1)
        return
link(ax)        
    
### Plotting the intersetion polygons
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
        x /= 1000
        y /= 1000
        ax.plot(x, y, c = color, alpha= alpha, zorder = zorder)
    return

polyg(ax)
    
### Axis, labels, ration, sizes, ...
ax.set_aspect("equal")
ax.set_title("Nodes + Links + Intersection Polygons", fontsize=10)
ax.set_xlabel("X [km]", fontsize=10)
ax.set_ylabel("Y [km]", fontsize=10)
ax.tick_params(axis='both', labelsize=8)

fig.savefig("graph.png")
plt.close(fig)



def numerical_sort(value):
    """
    This function sorts files in the correct numerical order
    """
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1
   

cmap = plt.get_cmap("coolwarm")

# Dl._vdist_3min[simulation number, timestamp, link id]
# DL.segment_lengths[link id]

def gradient_gif(param: list, param_name:list, fps : int):
    """
    Creates a gif made of the graphs ... for each parameter
    param : list of parameters 
    fps : amount of images per second (higher fps = shorter gif)
    """
    
    i = -1 
    
    for pn in param_name:
        # We create a folder for each parameter to store our png
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", str(pn)), exist_ok=True)    
        
    for p in param:
        i += 1
        
        for t in range(p.shape[1]):
            # We create a graph for each time t, for each parameter p
            fig, ax = plt.subplots(dpi = 250)

            ### Plotting nodes
            ax.scatter(DL.node_coordinates[:,0]/1000, DL.node_coordinates[:,1]/1000, s=10, c = "black", alpha=0.5, zorder = -2)
            
        
            # Normalized color gradient based on the min and max values of p
            norm = mcolors.Normalize(
                np.nanmin(p[0,:]),
                np.nanmax(p[0,:])
            )
            
            # Chosing the colors
            cmap = cm.get_cmap("coolwarm")

            ### Plotting the links
            link(ax, grad=True, norm = norm, p = p, t = t, cmap = cmap)

            ### Plotting the intersetion polygons
            polyg(ax, "black", 1,1)
                
                
            ### Axis, labels, ratio, sizes, ...
            ax.set_aspect("equal")
            ax.set_title(f"{param_name[i]} @ {fps} fps : t = {t}", fontsize=10)
            ax.set_xlabel("X [m]", fontsize=10)
            ax.set_ylabel("Y [m]", fontsize=10)
            ax.tick_params(axis='both', labelsize=8)

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array(p[0, t, :])
            fig.colorbar(sm, ax=ax, label=str(param_name[i]), location="right")

            fig.savefig(f"figure/{param_name[i]}/graph{str(t)}.png")
            print(f"t = {t}")
            plt.close(fig)
        
        ### We take the pngs created and sort them with the previous function numerical_sort()    
        files = sorted([f for f in os.listdir(f"figure/{param_name[i]}") if f.endswith(".png")])
        files = sorted(files, key=numerical_sort)
        
        ### We create now a gif with the pngs
        images = []
        for filename in files:
            print(filename)
            filepath = os.path.join(f"figure/{param_name[i]}", filename)
            images.append(imageio.imread(filepath))

        imageio.mimsave(os.path.join(f"gif/{param_name[i]}.gif"), images, fps=fps)
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