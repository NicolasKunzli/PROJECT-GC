import os
import json
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
figure_path = os.path.join(path, "figure")
os.makedirs(figure_path, exist_ok=True)

centroid = pd.read_csv(os.path.join(path, "metadata", "centroid_pos.csv"))
connections = pd.read_csv(os.path.join(path, "metadata", "link_bboxes_clustered.csv"))
with open(os.path.join(path, "metadata", "sections_of_interest.txt"), "r") as f:interest = [line.strip() for line in f.readlines()]


plt.figure(figsize=(10, 10), dpi = 100)
for id in interest:
    id

for _, row in centroid.iterrows():
    plt.scatter(centroid["x"], centroid["y"], s=40, c = "blue")

for _, row in connections.iterrows():
    x_vals = [row["from_x"], row["to_x"]]
    y_vals = [row["from_y"], row["to_y"]]
    plt.plot(x_vals, y_vals, c="red", linewidth = row["num_lanes"]/2)


plt.xlabel("X")
plt.ylabel("Y")
plt.title("")
plt.axis("equal")
plt.savefig(path+"/figure/map")
#plt.show()

#Visualisation of the simulations data

df = pd.read_pickle(path + "/simulation_sessions/session_000/timeseries/agg_timeseries.pkl")

print(df.keys())


#Visualisation of the polygons  
json_path = os.path.join(path, "metadata", "intersec_polygon.json")

with open(json_path, "r") as f:
    polygons = json.load(f)

plt.figure(figsize=(10, 10))

for section_id, data in polygons.items():
    poly = data["polygon"]
    
    # Extract x and y
    x = [p[0] for p in poly] + [poly[0][0]]  # close polygon
    y = [p[1] for p in poly] + [poly[0][1]]
    
    plt.plot(x, y)

plt.gca().set_aspect("equal")
plt.title("Intersection Polygons")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
