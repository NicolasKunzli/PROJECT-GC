import os
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
figure_path = os.path.join(path, "figure")
os.makedirs(figure_path, exist_ok=True)

centroid = pd.read_csv(path+"\metadata\centroid_pos.csv")
connections = pd.read_csv(path+"\metadata\link_bboxes_clustered.csv")
interest = pd.read_csv(path+"\metadata\sections_of_interest.txt")


plt.figure(figsize=(10, 10), dpi = 100)
for id in interest:
    print(id)
    
    
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
plt.show()