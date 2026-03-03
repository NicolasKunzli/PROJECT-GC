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

os.makedirs(figure_path, exist_ok=True)
os.makedirs(localfigure, exist_ok=True)

centroid = pd.read_csv(os.path.join(path, "metadata", "centroid_pos.csv"))
links = pd.read_csv(os.path.join(path, "metadata", "link_bboxes.csv"))
connections = pd.read_csv(os.path.join(path, "metadata", "connections.csv"))

polygons = pd.read_json(os.path.join(path, "metadata", "intersec_polygon.json"))
lane_info = pd.read_json(os.path.join(path, "metadata", "lane_info.json"))
od = pd.read_json(os.path.join(path, "metadata", "od_pairs.json"))
with open (os.path.join(path, "metadata", "train_test_split.json")) as f:tts=f.read()

with open(os.path.join(path, "metadata", "sections_of_interest.txt"), "r") as f:interest = [line.strip() for line in f.readlines()]

""" print("")
print(f"The keys of the file centroid_pos.csv : {centroid.keys()}")
print("")
print(f"The keys of the file connections.csv : {connections.keys()}")
print("")
print(f"The keys of the file link_bboxes.csv : {links.keys()}")
print("")
print(f"The keys of the file intersec_polygon.json : {polygons.keys()}")
print("")
print(f"The keys of the file lane_info.json : {lane_info.keys()}")
print("")
print(f"The keys of the file od_pairs.json : {od.keys()}")
print("") """


""" ce = [np.min(centroid["id"]), np.max(centroid["id"])]
print(f"Centroid id range : {ce}")

li = [np.min(links["id"]),np.max(links["id"])]
print(f"Links id range : {li}")

print(od) """



### Instance of the class
DL = DataLoader()
DL.init_graph_structure



### Creating the figure
fig, ax = plt.subplots(dpi = 250)

### Plotting nodes
ax.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=10, c = "black", alpha=1, zorder = 1)

### Plotting the links
for i, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]     
    ax.plot(x, y, c="red", linewidth=1, zorder = 2)
    
    
### Plotting the intersetion polygons
for section_id, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    x = [p[0] for p in poly] + [poly[0][0]]
    y = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(x, y, c = "blue", alpha = 1, zorder= 3)
    
### Axis, labels, ration, sizes, ...
ax.set_aspect("equal")
ax.set_title("Nodes + Links + Intersection Polygons")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

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
            ax.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=10, c = "black", alpha=0.5, zorder = -2)
            
            # Normalized color gradient based on the min and max values of p
            norm = mcolors.Normalize(
                np.nanmin(p[0,:]),
                np.nanmax(p[0,:])
            )
            # Chosing the colors
            cmap = cm.get_cmap("coolwarm")

            ### Plotting the links
            for j, row in links.iterrows():
                x = [row["from_x"], row["to_x"]]
                y = [row["from_y"], row["to_y"]]  
                if pd.isna(p[0,t,j]):  
                    z = "orange"
                else:
                    z = cmap(norm(p[0,t,j]))
                ax.plot(x, y, c=z, linewidth=1)

            ### Plotting the intersetion polygons
            for section_id, data in DL.intersection_polygon.items():
                poly = data["polygon"]
                x = [p[0] for p in poly] + [poly[0][0]]
                y = [p[1] for p in poly] + [poly[0][1]]
                ax.plot(x, y, c = "purple", alpha = 0.5, zorder= -1)
                
            ### Axis, labels, ratio, sizes, ...
            ax.set_aspect("equal")
            ax.set_title(f"{param_name[i]} @ {fps} fps : T = {t}")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array(p[0, t, :])
            fig.colorbar(sm, ax=ax, label=str(param_name[i]))

            fig.savefig(f"figure/{param_name[i]}/graph{str(t)}.png")
            print(f"T = {t}")
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

        imageio.mimsave(os.path.join(f"{param_name[i]}.gif"), images, fps=fps)

param = [DL._vdist_3min, DL._vtime_3min]
param_name = ["vdist_3min", "vtime_3min"]
fps = 0.5
gradient_gif(param, param_name, fps)