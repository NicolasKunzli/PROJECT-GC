"""
config.py — Shared constants, paths, metadata, and the DataLoader singleton.

Every module that needs DL, links, connections, or output paths imports from here.
This file is loaded once; Python's module cache ensures DL is instantiated only once
regardless of how many modules import it.
"""

import os
import json

import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib
matplotlib.use("Agg")

from DataLoad import DataLoader


#  Color used for segments flagged as "no data" (Case C in the NaN policy) 
NODATA_COLOR = "#cccccc"

#  Paths 
DATA_ROOT   = os.path.join(os.path.expanduser("~"), "Documents", "simbarca_upload")
FIGURE_PATH = os.path.join(DATA_ROOT, "figure")
LOCAL_FIGURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")
LOCAL_GIF    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")

os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(LOCAL_FIGURE, exist_ok=True)
os.makedirs(LOCAL_GIF,    exist_ok=True)

#  Metadata DataFrames 
_meta = os.path.join(DATA_ROOT, "metadata")

centroid    = pd.read_csv(os.path.join(_meta, "centroid_pos.csv"))
links       = pd.read_csv(os.path.join(_meta, "link_bboxes.csv")).sort_values("id").reset_index(drop=True)
connections = pd.read_csv(os.path.join(_meta, "connections.csv"))
polygons    = pd.read_json(os.path.join(_meta, "intersec_polygon.json"))
lane_info   = pd.read_json(os.path.join(_meta, "lane_info.json"))
od          = pd.read_json(os.path.join(_meta, "od_pairs.json"))

with open(os.path.join(_meta, "train_test_split.json")) as f:
    tts = f.read()
with open(os.path.join(_meta, "sections_of_interest.txt"), "r") as f:
    interest = [line.strip() for line in f.readlines()]

# Link centre-points (used by clustering feature builders and draw functions)
links["c_x"] = (links["from_x"] + links["to_x"]) / 2
links["c_y"] = (links["from_y"] + links["to_y"]) / 2

#  DataLoader singleton 
# DL._vdist_3min[session, timestamp, link_id]
# DL.segment_lengths[link_id]
DL = DataLoader()

#  Network connectivity (sparse, for Ward clustering) 
NETWORK_CONNECTIVITY = sparse.csr_matrix(DL.adjacency)
