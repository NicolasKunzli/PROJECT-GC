import pandas as pd
import matplotlib.pyplot as plt




path = "metadata/"
centroid = pd.read_csv(path+"centroid_pos.csv")
connections = pd.read_csv(path+"link_bboxes_clustered.csv")

positions = {
    row["id"]: (row["x"], row["y"])
    for _, row in centroid.iterrows()
}

plt.figure(figsize=(10, 10))

plt.scatter(centroid["x"], centroid["y"], s=40)

for _, row in connections.iterrows():
    x_vals = [row["from_x"], row["to_x"]]
    y_vals = [row["from_y"], row["to_y"]]
    plt.plot(x_vals, y_vals, c="red", linewidth = 2)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Graph du réseau routier")
plt.axis("equal")  # garder proportions correctes
plt.savefig("figure/map")
plt.show()