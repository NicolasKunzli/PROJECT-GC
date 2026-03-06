"""
analysis.py  –  Speed analysis, data-reliability diagnostics & congestion wave visualisation
Barcelona (SimBArCa) traffic dataset
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import pandas as pd
from scipy.ndimage import uniform_filter1d          # for smoothing shockwave overlay

from DataLoad import DataLoader

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
FIG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", "analysis")
LINKS_CSV = os.path.join(BASE, "metadata", "link_bboxes.csv")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, "spacetime"), exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
DL    = DataLoader()
links = pd.read_csv(LINKS_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPUTE SPEED  (sessions × T × N)  →  average across ALL sessions
# ─────────────────────────────────────────────────────────────────────────────
speed = np.divide(
    DL._vdist_3min,
    DL._vtime_3min,
    out=np.full_like(DL._vdist_3min, np.nan),
    where=DL._vtime_3min != 0,
)
# shape (T, N) — session-averaged
speed_all = np.nanmean(speed, axis=0)

# Real timestamps for axis labels (3-min steps between 08:00 and 10:00)
T = speed_all.shape[0]
timestamps = pd.date_range("2005-05-10 08:03", periods=T, freq="3min")

def fmt_time(x, _):
    """Format a float tick as HH:MM."""
    i = int(np.clip(x, 0, T - 1))
    return timestamps[i].strftime("%H:%M")

# ─────────────────────────────────────────────────────────────────────────────
# 2. AVERAGE SPEED MAP  (all sessions)
# ─────────────────────────────────────────────────────────────────────────────
mean_speed = np.nanmean(speed_all, axis=0)   # (N,)

fig, ax = plt.subplots(dpi=250)
norm_sp = mcolors.Normalize(np.nanmin(mean_speed), np.nanmax(mean_speed))
cmap_sp = cm.get_cmap("RdYlGn")

for _, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    idx = DL.section_id_to_index.get(row["id"])
    c   = cmap_sp(norm_sp(mean_speed[idx])) if idx is not None else "grey"
    ax.plot(x, y, c=c, linewidth=1.2)

for _, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, c="grey", alpha=0.4, lw=0.7, zorder=-1)

sm = cm.ScalarMappable(norm=norm_sp, cmap=cmap_sp)
sm.set_array(mean_speed)
fig.colorbar(sm, ax=ax, label="Mean speed (m/s)")
ax.set_aspect("equal")
ax.set_title(f"Average Speed Map  (all {DL.num_sessions} sessions)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
out = os.path.join(FIG_DIR, "avg_speed_map.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 20 SLOWEST SEGMENTS  (highlighted on map)
# ─────────────────────────────────────────────────────────────────────────────
slow_idx = set(np.argsort(mean_speed)[:20])   # speed-array indices
first_red = True

fig, ax = plt.subplots(dpi=250)
for _, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    idx = DL.section_id_to_index.get(row["id"])
    if idx is not None and idx in slow_idx:
        lbl = "Slowest 20" if first_red else ""
        ax.plot(x, y, c="red", linewidth=2.5, zorder=2, label=lbl)
        ax.scatter([(x[0]+x[1])/2], [(y[0]+y[1])/2], s=30, c="darkred", zorder=3)
        first_red = False
    else:
        ax.plot(x, y, c="lightgrey", linewidth=0.8, zorder=1)

for _, data in DL.intersection_polygon.items():
    poly = data["polygon"]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, c="grey", alpha=0.3, lw=0.7, zorder=-1)

ax.set_aspect("equal")
ax.set_title(f"20 Slowest Road Segments  (all {DL.num_sessions} sessions, time-averaged)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.legend(loc="upper right")
out = os.path.join(FIG_DIR, "slowest_20_segments.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NETWORK SPEED vs TIME  (all sessions + per-session faint lines)
# ─────────────────────────────────────────────────────────────────────────────
network_speed = np.nanmean(speed_all, axis=1)   # (T,)
t_axis        = np.arange(T)

fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
for s in range(speed.shape[0]):
    sess_speed = np.nanmean(speed[s], axis=1)
    ax.plot(t_axis, sess_speed, color="steelblue", linewidth=0.6, alpha=0.2)
ax.plot(t_axis, network_speed, color="steelblue", linewidth=2.0, label="All-session mean")
ax.fill_between(t_axis, network_speed, alpha=0.12, color="steelblue")
ax.xaxis.set_major_formatter(FuncFormatter(fmt_time))
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title(f"Network-Average Speed over Time  (all {DL.num_sessions} sessions)")
ax.set_xlabel("Time"); ax.set_ylabel("Mean speed (m/s)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
out = os.path.join(FIG_DIR, "network_speed_vs_time.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────
nan_frac      = np.mean(np.isnan(speed_all), axis=0)   # (N,)
high_nan_mask = nan_frac > 0.5
n_high        = high_nan_mask.sum()
print(f"\n  Data reliability:")
print(f"    Segments with >50 % missing data : {n_high}  /  {len(nan_frac)}")
print(f"    Overall NaN rate                 : {nan_frac.mean()*100:.1f} %")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
norm_nan = mcolors.Normalize(0, 1)
cmap_nan = cm.get_cmap("hot_r")

ax = axes[0]
for _, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    idx = DL.section_id_to_index.get(row["id"])
    c   = cmap_nan(norm_nan(nan_frac[idx])) if idx is not None else "grey"
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
ax.set_title("Sensor Coverage Map\n(dark = more missing)")
ax.set_xlabel("X"); ax.set_ylabel("Y")

ax = axes[1]
first_crimson = True
for _, row in links.iterrows():
    x = [row["from_x"], row["to_x"]]
    y = [row["from_y"], row["to_y"]]
    idx = DL.section_id_to_index.get(row["id"])
    if idx is not None and high_nan_mask[idx]:
        lbl = ">50 % missing" if first_crimson else ""
        ax.plot(x, y, c="crimson", linewidth=2.0, zorder=2, label=lbl)
        first_crimson = False
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

fig.suptitle(f"Data Reliability Analysis  (all {DL.num_sessions} sessions)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(FIG_DIR, "data_reliability_map.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

# NaN distribution histogram
# Use per-session nan fractions (shape S×N) so every session contributes
# independently — gives a richer, more accurate picture than the session-average alone
nan_frac_per_session = np.mean(np.isnan(speed), axis=1)   # (S, N): NaN rate per session per segment
nan_frac_flat        = nan_frac_per_session.flatten()      # (S*N,)

fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
ax.hist(nan_frac_flat, bins=40, color="steelblue", edgecolor="white",
        label=f"All sessions (n={speed.shape[0]}×{speed.shape[2]} obs)")
ax.axvline(0.5, color="crimson", linestyle="--", linewidth=1.5, label=">50 % threshold")
ax.axvline(nan_frac_flat.mean(), color="orange", linestyle=":", linewidth=1.5,
           label=f"Mean = {nan_frac_flat.mean()*100:.1f} %")
ax.set_xlabel("Fraction of time steps with missing speed  (per segment per session)")
ax.set_ylabel("Count  (segment × session pairs)")
ax.set_title(f"Distribution of Missing Data per Segment\n"
             f"Barcelona SimBArCa  |  all {DL.num_sessions} sessions")
ax.legend()
out = os.path.join(FIG_DIR, "nan_distribution.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

# Spatial autocorrelation of NaNs
adj = DL.adjacency
neighbour_nan = []
for i in range(len(nan_frac)):
    nbrs = np.where(adj[i] == 1)[0]
    neighbour_nan.append(np.mean(nan_frac[nbrs]) if len(nbrs) else np.nan)
neighbour_nan = np.array(neighbour_nan)
valid = ~np.isnan(neighbour_nan)
corr  = np.corrcoef(nan_frac[valid], neighbour_nan[valid])[0, 1]
print(f"\n  Spatial NaN clustering: r = {corr:.3f}")
print(f"    {'NaNs ARE spatially clustered.' if corr > 0.3 else 'No strong spatial clustering.'}")

fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
ax.scatter(nan_frac[valid], neighbour_nan[valid], s=8, alpha=0.5, color="steelblue")
m, b = np.polyfit(nan_frac[valid], neighbour_nan[valid], 1)
xs   = np.linspace(0, 1, 100)
ax.plot(xs, m*xs + b, color="crimson", linewidth=1.5, label=f"r = {corr:.3f}")
ax.set_xlabel("Segment NaN fraction"); ax.set_ylabel("Mean NaN fraction of neighbours")
ax.set_title("Spatial Clustering of Missing Data"); ax.legend()
out = os.path.join(FIG_DIR, "nan_spatial_clustering.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")


# =============================================================================
# 6. CONGESTION WAVE ANALYSIS  –  Space-Time Diagrams
#    Barcelona SimBArCa dataset  |  08:03 – 10:00  |  3-min intervals
# =============================================================================
print("\n─── Congestion wave analysis ───")

# ── 6a. Helper: free-flow speed & congestion index ───────────────────────────
# Free-flow = 85th-percentile speed per segment across all sessions & time
#   (more robust than max which is noise-sensitive)
speed_flat = speed.reshape(-1, speed.shape[-1])          # (S*T, N)
free_flow  = np.nanpercentile(speed_flat, 85, axis=0)    # (N,)
free_flow  = np.where(free_flow == 0, np.nan, free_flow) # avoid /0

# Congestion index CI ∈ [0, 1]:  0 = free flow,  1 = fully stopped
#   per session (S, T, N) for richer visualisation
ci = 1.0 - speed / free_flow[np.newaxis, np.newaxis, :]
ci = np.clip(ci, 0, 1)

# Session-averaged CI  (T, N)
ci_all = np.nanmean(ci, axis=0)

# ── 6b. Identify the 20 slowest corridor indices (already computed) ──────────
slow_idx_list = list(np.argsort(mean_speed)[:20])   # ordered slowest → faster

# Section-id labels for axis ticks
slow_section_ids = [DL.index_to_section_id.get(i, i) for i in slow_idx_list]

# ── 6c. Build corridor matrix  (T × 20)  for the space-time diagram ────────── 
ci_corridor    = ci_all[:, slow_idx_list]             # (T, 20)
speed_corridor = speed_all[:, slow_idx_list]          # (T, 20) — for second diagram
tick_step      = max(1, T // 10)                      # x-axis tick spacing

# ── 6e. PLOT 2 — Space-Time Speed heatmap (m/s) ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), dpi=220)

vmin_sp = np.nanpercentile(speed_corridor, 2)
vmax_sp = np.nanpercentile(speed_corridor, 98)

im = ax.imshow(
    speed_corridor.T,
    aspect="auto",
    origin="lower",
    cmap="RdYlGn",
    vmin=vmin_sp, vmax=vmax_sp,
    interpolation="nearest",
)
'''# shockwave contour at 0.5 × free-flow threshold
ff_corridor = free_flow[slow_idx_list]                     # (20,)
threshold   = 0.5 * ff_corridor[:, np.newaxis]             # (20, 1)
shock_mask  = (speed_corridor.T < threshold).astype(float) # (20, T)
smoothed_s  = uniform_filter1d(shock_mask, size=2, axis=1)
ax.contour(smoothed_s, levels=[0.5], colors="white", linewidths=1.0,
          linestyles="-", alpha=0.85)'''

fig.colorbar(im, ax=ax, label="Speed  (m/s)")

ax.set_xticks(range(0, T, tick_step))
ax.set_xticklabels([timestamps[i].strftime("%H:%M") for i in range(0, T, tick_step)],
                   rotation=30, ha="right")
ax.set_yticks(range(20))
ax.set_yticklabels([f"Seg {sid}" for sid in slow_section_ids], fontsize=7)

ax.set_xlabel("Time  (08:00 → 10:00,  3-min intervals)")
ax.set_ylabel("Road Segment  (slowest 20, ranked)")
ax.set_title(
    f"Space-Time Speed Diagram — 20 Slowest Corridors\n"
    f"Barcelona SimBArCa  |  all {DL.num_sessions} sessions averaged  |  "
#    f"white contour = speed < 50 % free-flow (shockwave front)"
)
fig.tight_layout()
out = os.path.join(FIG_DIR, "spacetime", "speed_spacetime.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

# ── 6h. PLOT 5 — Network-level congestion share over time ────────────────────
# What fraction of ALL network segments exceed CI > 0.4 at each timestep?
THRESH = 0.4
congestion_share = np.nanmean(ci_all > THRESH, axis=1)   # (T,)

fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
ax.fill_between(np.arange(T), congestion_share * 100, alpha=0.3, color="tomato")
ax.plot(np.arange(T), congestion_share * 100, color="crimson", linewidth=1.8)
ax.axhline(50, color="grey", linestyle="--", linewidth=0.9, label="50 % of network congested")

ax.xaxis.set_major_formatter(FuncFormatter(fmt_time))
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_ylim(0, 100)
ax.set_xlabel("Time  (08:00 → 10:00)")
ax.set_ylabel("% of network segments with CI > 0.40")
ax.set_title(
    f"Network-Wide Congestion Share over Time\n"
    f"Barcelona SimBArCa  |  all {DL.num_sessions} sessions averaged"
)
ax.legend(); ax.grid(True, alpha=0.25)
fig.tight_layout()
out = os.path.join(FIG_DIR, "spacetime", "network_congestion_share.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"  ✓ saved {out}")

print(f"\nAll done — figures in: {FIG_DIR}")
print(f"  Speed & reliability : {FIG_DIR}/*.png")
print(f"  Congestion waves    : {FIG_DIR}/spacetime/*.png")