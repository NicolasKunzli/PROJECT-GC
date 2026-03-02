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
ax.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=10, c = "black", alpha=0.5, zorder = -2)

""" for i in range(DL.adjacency.shape[0]):
    for j in range(i+1, DL.adjacency.shape[1]): #we only take the upper triangular part since the matrix is symetric
        if DL.adjacency[i,j] == 1:
            x = [DL.node_coordinates[i,0], DL.node_coordinates[j,0]]
            y = [DL.node_coordinates[i,1], DL.node_coordinates[j,1]]

            plt.plot(x, y, c='red', linewidth=(DL.num_lanes[i] + DL.num_lanes[j])/4) """

norm = mcolors.Normalize(
    np.nanmin(DL._vdist_3min[0,:]),
    np.nanmax(DL._vdist_3min[0,:])
)

cmap = cm.get_cmap("coolwarm")

### Plotting the links
for i, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]     
    z = cmap(norm(DL._vdist_3min[0,0,i]))
    ax.plot(x, y, c=z, linewidth=1)

### Plotting the intersetion polygons
for section_id, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    x = [p[0] for p in poly] + [poly[0][0]]
    y = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(x, y, c = "grey", alpha = 0.5, zorder= -1)
    
### Axis, labels, ration, sizes, ...
ax.set_aspect("equal")
ax.set_title("Nodes + Links + Intersection Polygons")
ax.set_xlabel("X")
ax.set_ylabel("Y")

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(DL._vdist_3min[0, 0, :])
fig.colorbar(sm, ax=ax, label="vdist (3min)")

fig.savefig("graph.png")
plt.close(fig)




def numerical_sort(value):

    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1









param1 = ["vdist_3min", "vtime_3min"]
for p1 in param1:
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", str(p1)), exist_ok=True)
    
i = -1    
param = [DL._vdist_3min, DL._vtime_3min]
for p in param:
    i += 1
    for t in range(p.shape[1]):

        fig, ax = plt.subplots(dpi = 250)

        ### Plotting nodes
        ax.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=10, c = "black", alpha=0.5, zorder = -2)

        norm = mcolors.Normalize(
            np.nanmin(p[0,:]),
            np.nanmax(p[0,:])
        )

        cmap = cm.get_cmap("coolwarm")

        ### Plotting the links
        for j, row in links.iterrows():
            x = [row["from_x"], row["to_x"]]
            y = [row["from_y"], row["to_y"]]     
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
        ax.set_title(f"{param1[i]} : T = {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(p[0, t, :])
        fig.colorbar(sm, ax=ax, label=str(param1[i]))

        fig.savefig(f"figure/{param1[i]}/graph{str(t)}.png")
        print(f"T = {t}")
        plt.close(fig)
        
    files = sorted([f for f in os.listdir(f"figure/{param1[i]}") if f.endswith(".png")])
    files = sorted(files, key=numerical_sort)
    
    images = []
    for filename in files:
        print(filename)
        filepath = os.path.join(f"figure/{param1[i]}", filename)
        images.append(imageio.imread(filepath))

    imageio.mimsave(os.path.join(f"{param1[i]}.gif"), images, fps=5)