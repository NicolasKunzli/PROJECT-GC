import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
from DataLoad import DataLoader

path = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
figure_path = os.path.join(path, "figure")
os.makedirs(figure_path, exist_ok=True)

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

DL = DataLoader()
DL.init_graph_structure

print(DL.node_coordinates.shape)
print("")
print(DL.num_lanes.shape)
print("")
print(DL.adjacency.shape)
print("")
print(DL.segment_lengths.shape)
print("")
print(len(DL.section_id_to_index))


plt.figure(figsize=(10, 10), dpi = 250)

plt.scatter(DL.node_coordinates[:,0], DL.node_coordinates[:,1], s=15, c = "blue")

""" for i in range(DL.adjacency.shape[0]):
    for j in range(i+1, DL.adjacency.shape[1]): #we only take the upper triangular part since the matrix is symetric
        if DL.adjacency[i,j] == 1:
            x = [DL.node_coordinates[i,0], DL.node_coordinates[j,0]]
            y = [DL.node_coordinates[i,1], DL.node_coordinates[j,1]]

            plt.plot(x, y, c='red', linewidth=(DL.num_lanes[i] + DL.num_lanes[j])/4) """
            
DL.link_bboxes = pd.read_csv(os.path.join(DL.metadata_folder, "link_bboxes_clustered.csv"))     

for _, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]     
    plt.plot(x, y, c='red', linewidth=1)

for section_id, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    x = [p[0] for p in poly] + [poly[0][0]]
    y = [p[1] for p in poly] + [poly[0][1]]
    plt.plot(x, y, c = "black")
    
plt.plot()
plt.gca().set_aspect("equal")
plt.title("Nodes + Links + Intersection Polygons")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("graph.png")
plt.close()