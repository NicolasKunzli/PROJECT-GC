"""
generate_report_figures.py — one-shot script to produce all figures needed by report.tex.

Figures generated:
  figure/graph_search/norm_q0.52/merged_graph_search_analysis.png
  figure/graph_search/raw_q2/merged_graph_search_analysis.png
  figure/clustering/grid_clusters/grid_kmeans5_qc0.520.png
  figure/clustering/grid_clusters/grid_kmeans9_qc0.520.png
  figure/clustering/grid_clusters/grid_kmeans5_qc0.520_median.png
"""

import os
import numpy as np
from config import LOCAL_FIGURE

QC = 0.52
TIMESTEPS = [0, 3, 5, 7, 9, 13, 17, 19, 21, 23, 25, 29, 33, 36, 38]
KEY_TS    = [0, 7, 19, 38]

print("=" * 60)
print("Generating report figures")
print("=" * 60)

# ── Approach 1: BFS graph-search analysis ────────────────────────────────────
print("\n[1/2] BFS analysis — normalised speed (qc=0.52)...")
from percolation.graph_search import run_graph_search_analysis

run_graph_search_analysis(
    qc=QC, use_normalized=True, key_timesteps=KEY_TS,
    folder=os.path.join(LOCAL_FIGURE, "graph_search", "norm_q0.52"),
)

print("\n[2/4] BFS analysis — raw speed (q=2.0 m/s)...")
run_graph_search_analysis(
    qc=2.0, use_normalized=False, key_timesteps=KEY_TS,
    folder=os.path.join(LOCAL_FIGURE, "graph_search", "raw_q2"),
)

# ── Approach 2: Grid K-means ─────────────────────────────────────────────────
print("\n[3/4] Grid K-means k=5 (spatial map + median plot)...")
from percolation.maps import grid_clust_kmeans, grid_clust_kmeans_temporal

grid_clust_kmeans(xdiv=20, ydiv=16, k=5, qc=QC)
grid_clust_kmeans_temporal(xdiv=20, ydiv=16, k=5, qc=QC, timesteps=TIMESTEPS)

print("\n[4/4] Grid K-means k=9 (spatial map)...")
grid_clust_kmeans(xdiv=20, ydiv=16, k=9, qc=QC)

# ── Print numeric data for tables ────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 1 — Approach 1 (BFS, qc=0.52) metrics")
print("=" * 60)

def t_label(t):
    total = 8 * 60 + 3 + t * 3
    return f"{total // 60:02d}:{total % 60:02d}"

n = result_norm
print(f"{'Time':>6}  {'N':>3}  {'S1':>4}  {'S2':>4}  {'S3':>4}  "
      f"{'v1':>5}  {'v2':>5}  {'v3':>5}  {'vbg':>5}")
for i, t in enumerate(TIMESTEPS):
    N  = n["n_clusters"][i]
    s  = n["sizes"][i]
    m  = n["medians"][i]
    bg = n["bg_median"][i]
    print(f"{t_label(t):>6}  {N:>3}  "
          f"{s[0]:>4.0f}  {s[1]:>4.0f}  {s[2]:>4.0f}  "
          f"{m[0]:>5.2f}  {m[1]:>5.2f}  {m[2]:>5.2f}  {bg:>5.2f}")

print("\n" + "=" * 60)
print("TABLE 1 (raw) — Approach 1 (BFS, q=2.0 m/s) metrics")
print("=" * 60)
r = result_raw
print(f"{'Time':>6}  {'N':>3}  {'S1':>4}  {'S2':>4}  {'S3':>4}  "
      f"{'v1':>5}  {'v2':>5}  {'v3':>5}  {'vbg':>5}")
for i, t in enumerate(TIMESTEPS):
    N  = r["n_clusters"][i]
    s  = r["sizes"][i]
    m  = r["medians"][i]
    bg = r["bg_median"][i]
    print(f"{t_label(t):>6}  {N:>3}  "
          f"{s[0]:>4.0f}  {s[1]:>4.0f}  {s[2]:>4.0f}  "
          f"{m[0]:>5.2f}  {m[1]:>5.2f}  {m[2]:>5.2f}  {bg:>5.2f}")

print("\nDone. Check figure/ for all outputs.")
