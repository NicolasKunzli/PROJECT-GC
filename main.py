import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
figure_path = os.path.join(path, "figure")
os.makedirs(figure_path, exist_ok=True)

centroid = pd.read_csv(os.path.join(path, "metadata", "centroid_pos.csv"))
connections = pd.read_csv(os.path.join(path, "metadata", "link_bboxes_clustered.csv"))
polygons = pd.read_json(os.path.join(path, "metadata", "intersec_polygon.json"))
with open(os.path.join(path, "metadata", "sections_of_interest.txt"), "r") as f:interest = [line.strip() for line in f.readlines()]




    
def centr():
    for _, row in centroid.iterrows():
        plt.scatter(centroid["x"], centroid["y"], s=40, c = "blue")

def conn():
    for _, row in connections.iterrows():
        x_vals = [row["from_x"], row["to_x"]]
        y_vals = [row["from_y"], row["to_y"]]
        plt.plot(x_vals, y_vals, c="red", linewidth = row["num_lanes"]/2)
        
def poly():   
    for section_id, data in polygons.items():
        poly = data["polygon"]
    
        # Extract x and y
        x = [p[0] for p in poly] + [poly[0][0]]  # close polygon
        y = [p[1] for p in poly] + [poly[0][1]]
    
        plt.plot(x, y, c = "green")
        
        
plt.figure(figsize=(10, 10), dpi = 100)
centr()
conn()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("")
plt.axis("equal")
plt.savefig(path+"/figure/map")
#plt.show()
plt.close()


#Visualisation of the simulations data

df = pd.read_pickle(path + "/simulation_sessions/session_000/timeseries/agg_timeseries.pkl")

print(df.keys())


plt.figure(figsize=(10, 10))
poly()
plt.gca().set_aspect("equal")
plt.title("Intersection Polygons")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(path+"/figure/polygons")
#plt.show()
plt.close()

plt.figure(figsize=(10, 10))
centr()
conn()
poly()
plt.gca().set_aspect("equal")
plt.title("")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(path+"/figure/mappolygons")
plt.show()
