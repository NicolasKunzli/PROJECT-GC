"""
analysis.py  –  Speed analysis + data-reliability diagnostics
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

from DataLoad import DataLoader

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
FIG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", "analysis")
LINKS_CSV = os.path.join(BASE, "metadata", "link_bboxes.csv")
os.makedirs(FIG_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
DL    = DataLoader()
links = pd.read_csv(LINKS_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPUTE SPEED  (vdist / vtime, shape: sessions × T × locations)
# ─────────────────────────────────────────────────────────────────────────────
speed = np.divide(
    DL._vdist_3min,
    DL._vtime_3min,
    out=np.full_like(DL._vdist_3min, np.nan),
    where=DL._vtime_3min != 0,
)
# session 0 throughout, shape (T, N)
speed_s0 = speed[0]

# ─────────────────────────────────────────────────────────────────────────────
# 2. AVERAGE SPEED MAP
# ─────────────────────────────────────────────────────────────────────────────
mean_speed = np.nanmean(speed_s0, axis=0)          # (N,)

fig, ax = plt.subplots(dpi=250)
norm_sp  = mcolors.Normalize(np.nanmin(mean_speed), np.nanmax(mean_speed))
cmap_sp  = cm.get_cmap("RdYlGn")

for j, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    c = cmap_sp(norm_sp(mean_speed[j])) if j < len(mean_speed) else "grey"
    ax.plot(x, y, c=c, linewidth=1.2)

# intersection polygons
for _, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, c="grey", alpha=0.4, lw=0.7, zorder=-1)

sm = cm.ScalarMappable(norm=norm_sp, cmap=cmap_sp)
sm.set_array(mean_speed)
fig.colorbar(sm, ax=ax, label="Mean speed (m/s)")
ax.set_aspect("equal")
ax.set_title("Average Speed Map  (session 0)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
out = os.path.join(FIG_DIR, "avg_speed_map.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 20 SLOWEST SEGMENTS  (highlighted on map)
# ─────────────────────────────────────────────────────────────────────────────
slow_idx = np.argsort(mean_speed)[:20]             # indices of 20 slowest

fig, ax = plt.subplots(dpi=250)
for j, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    if j in slow_idx:
        ax.plot(x, y, c="red", linewidth=2.5, zorder=2, label="Slowest 20" if j == slow_idx[0] else "")
    else:
        ax.plot(x, y, c="lightgrey", linewidth=0.8, zorder=1)

for _, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, c="grey", alpha=0.3, lw=0.7, zorder=-1)

ax.scatter(
    DL.node_coordinates[slow_idx, 0],
    DL.node_coordinates[slow_idx, 1],
    s=30, c="darkred", zorder=3,
)
ax.set_aspect("equal")
ax.set_title("20 Slowest Road Segments  (session 0, time-averaged)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.legend(loc="upper right")
out = os.path.join(FIG_DIR, "slowest_20_segments.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NETWORK SPEED vs TIME
# ─────────────────────────────────────────────────────────────────────────────
network_speed = np.nanmean(speed_s0, axis=1)       # (T,)
t_axis        = np.arange(len(network_speed))

fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
ax.plot(t_axis, network_speed, color="steelblue", linewidth=1.5)
ax.fill_between(t_axis, network_speed, alpha=0.15, color="steelblue")
ax.set_title("Network-Average Speed over Time  (session 0)")
ax.set_xlabel("Time step (3-min intervals)")
ax.set_ylabel("Mean speed (m/s)")
ax.grid(True, alpha=0.3)
out = os.path.join(FIG_DIR, "network_speed_vs_time.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA RELIABILITY  –  NaN / missing-data analysis
# ─────────────────────────────────────────────────────────────────────────────

# 5a. Missing-data fraction per segment  (over time, session 0)
nan_frac = np.mean(np.isnan(speed_s0), axis=0)    # (N,) fraction in [0,1]

# 5b. Segments with >50 % missing data
high_nan_mask = nan_frac > 0.5
n_high        = high_nan_mask.sum()
print(f"\n  Data reliability:")
print(f"    Segments with >50 % missing data : {n_high}  /  {len(nan_frac)}")
print(f"    Overall NaN rate                 : {nan_frac.mean()*100:.1f} %")

# --- MAP: coverage / missing-data map ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)

# left panel – NaN fraction
norm_nan = mcolors.Normalize(0, 1)
cmap_nan = cm.get_cmap("hot_r")

ax = axes[0]
for j, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    c = cmap_nan(norm_nan(nan_frac[j])) if j < len(nan_frac) else "grey"
    ax.plot(x, y, c=c, linewidth=1.2)
for _, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, c="grey", alpha=0.3, lw=0.7, zorder=-1)
sm = cm.ScalarMappable(norm=norm_nan, cmap=cmap_nan)
sm.set_array(nan_frac)
fig.colorbar(sm, ax=ax, label="Fraction missing")
ax.set_aspect("equal")
ax.set_title("Sensor Coverage Map\n(bright = more missing)")
ax.set_xlabel("X"); ax.set_ylabel("Y")

# right panel – highlight >50 % missing
ax = axes[1]
for j, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    if j < len(high_nan_mask) and high_nan_mask[j]:
        ax.plot(x, y, c="crimson", linewidth=2.0, zorder=2,
                label=">50 % missing" if j == np.where(high_nan_mask)[0][0] else "")
    else:
        ax.plot(x, y, c="lightgrey", linewidth=0.7, zorder=1)
for _, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, c="grey", alpha=0.3, lw=0.7, zorder=-1)
ax.set_aspect("equal")
ax.set_title(f"Segments with >50 % Missing Data\n({n_high} segments highlighted in red)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
if n_high > 0:
    ax.legend(loc="upper right")

fig.suptitle("Data Reliability Analysis  (session 0)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(FIG_DIR, "data_reliability_map.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out}")

# 5c. Spatial clustering of NaNs  –  histogram + Moran-style neighbour check
fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
ax.hist(nan_frac, bins=40, color="steelblue", edgecolor="white")
ax.axvline(0.5, color="crimson", linestyle="--", label=">50 % threshold")
ax.set_xlabel("Fraction of time steps with missing speed")
ax.set_ylabel("Number of road segments")
ax.set_title("Distribution of Missing Data per Segment")
ax.legend()
out = os.path.join(FIG_DIR, "nan_distribution.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out}")

# Spatial autocorrelation proxy: compare NaN fraction of a node vs. its neighbours
adj = DL.adjacency                                 # (N, N) binary
neighbour_nan = []
for i in range(len(nan_frac)):
    nbrs = np.where(adj[i] == 1)[0]
    if len(nbrs):
        neighbour_nan.append(np.mean(nan_frac[nbrs]))
    else:
        neighbour_nan.append(np.nan)
neighbour_nan = np.array(neighbour_nan)

valid = ~np.isnan(neighbour_nan)
corr  = np.corrcoef(nan_frac[valid], neighbour_nan[valid])[0, 1]
print(f"\n  Spatial NaN clustering:")
print(f"    Pearson r(segment NaN, mean-neighbour NaN) = {corr:.3f}")
print(f"    {'NaNs ARE spatially clustered.' if corr > 0.3 else 'NaNs show little spatial clustering.'}")

fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
ax.scatter(nan_frac[valid], neighbour_nan[valid],
           s=8, alpha=0.5, color="steelblue")
m, b = np.polyfit(nan_frac[valid], neighbour_nan[valid], 1)
xs   = np.linspace(0, 1, 100)
ax.plot(xs, m*xs + b, color="crimson", linewidth=1.5, label=f"r = {corr:.3f}")
ax.set_xlabel("Segment NaN fraction")
ax.set_ylabel("Mean NaN fraction of neighbours")
ax.set_title("Spatial Clustering of Missing Data")
ax.legend()
out = os.path.join(FIG_DIR, "nan_spatial_clustering.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ saved {out}")

print("\nAll done — figures saved to:", FIG_DIR)
