import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

logger = logging.getLogger("default")
# https://github.com/Weijiang-Xiong/OpenSkyTraffic/blob/master/skytraffic/data/datasets/simbarca_base.py

class DataLoader():

    data_root = os.path.normpath(os.path.join(os.path.dirname(__file__),".", "data"))
    metadata_folder = os.path.join(data_root, "metadata")
    session_splits = os.path.join(metadata_folder, "train_test_split.json")
    session_folder_pattern = "simulation_sessions/session_*"
    _sample_start_time = np.datetime64("2005-05-10T08:00:00") # so this is the day set in the simulator
    _sample_end_time = np.datetime64("2005-05-10T10:00:00")
    #data_null_value = float("nan")

    def __init__(self, split="train"):
        self.split = split

        # data sequences along with timestamps 
        # the provided raw sequences are vehicle travel distance (vdist) and travel time (vtime) for all locations (road segments)
        # taggregated every 5 seconds and every 3 minutes, corresponding to high-frequency drones and low-frequency loop detectors
        self._timestamp_5s: np.ndarray # shape (T_high,)
        self._vdist_5s: np.ndarray # shape (num_sessions, T_high, num_locations)
        self._vtime_5s: np.ndarray # shape (num_sessions, T_high, num_locations)
        self._timestamp_3min: np.ndarray # shape (T_low,)
        self._ld_speed_3min: np.ndarray # shape (num_sessions, T_low, num_locations)
        self._vdist_3min: np.ndarray # shape (num_sessions, T_low, num_locations)
        self._vtime_3min: np.ndarray # shape (num_sessions, T_low, num_locations)
        
        # simulation session info, for grouped evaluation 
        self.session_ids: List[int] 
        
        # graph structure 
        self.adjacency: np.ndarray
        self.segment_lengths: np.ndarray
        self.edge_index: np.ndarray
        self.node_coordinates: np.ndarray
        self.section_ids_sorted: np.ndarray
        self.section_id_to_index: Dict[int, int]
        self.index_to_section_id: Dict[int, int]
        self.intersection_polygon: Dict[str, Any]

        # initialize data sequences and metadata 
        self.init_raw_sequences(self.load_seq_files())
        self.init_graph_structure()
        session_info = self.get_session_properties()
        self.session_ids = session_info["session_ids"]

    @property
    def num_sessions(self) -> int:
        return len(self.session_ids)

    def load_seq_files(self) -> List[Dict[str, Any]]:
        """Load agg_timeseries.pkl files from various session folders in parallel"""
        
        sessions_in_split = self.get_sessions_in_split()
        if not sessions_in_split:
            logger.warning("No sessions found for split '{}'".format(self.split))
            return []
            
        sample_files = [os.path.join(f, "timeseries", "agg_timeseries.pkl") for f in sessions_in_split]
        logger.info("Loading {} sample files for {} split in parallel".format(len(sample_files), self.split))
        
        def load_file(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        with ThreadPoolExecutor() as executor:
            loaded_seqs: List[pd.DataFrame] = list(executor.map(load_file, sample_files))

        return loaded_seqs

    def init_raw_sequences(self, seqs: List[pd.DataFrame]):
        # drone measurements, every 5s
        timestamp_5s: pd.DatetimeIndex = seqs[0]['drone_vdist'].index.to_numpy()
        start_index_5s = np.where(timestamp_5s == self._sample_start_time)[0][0] + 1 # do not include 8:00 
        end_index_5s = np.where(timestamp_5s == self._sample_end_time)[0][0] # do not include 10:00 
        self._timestamp_5s = timestamp_5s[start_index_5s:end_index_5s]
        self._vdist_5s = np.stack([seq['drone_vdist'].iloc[start_index_5s:end_index_5s].to_numpy() for seq in seqs], axis=0)
        self._vtime_5s = np.stack([seq['drone_vtime'].iloc[start_index_5s:end_index_5s].to_numpy() for seq in seqs], axis=0)

        # loop detector measurements, every 3 minutes 
        timestamp_3min: pd.DatetimeIndex = seqs[0]['pred_vtime'].index.to_numpy()
        start_index_3min = np.where(timestamp_3min == self._sample_start_time)[0][0] + 1 # do not include 8:00 
        end_index_3min = np.where(timestamp_3min == self._sample_end_time)[0][0] # do not include 10:00 
        self._timestamp_3min = timestamp_3min[start_index_3min:end_index_3min]
        self._ld_speed_3min = np.stack([seq['ld_speed'].iloc[start_index_3min:end_index_3min].to_numpy() for seq in seqs], axis=0)

        # sequences for constructing prediction targets, every 3 minutes 
        self._vdist_3min = np.stack([seq['pred_vdist'].iloc[start_index_3min:end_index_3min].to_numpy() for seq in seqs], axis=0)
        self._vtime_3min = np.stack([seq['pred_vtime'].iloc[start_index_3min:end_index_3min].to_numpy() for seq in seqs], axis=0)
        

    def clean_up_raw_sequences(self):
        # this is for cleaning up the raw sequences after obtaining required ones in a subclass
        del self._vdist_5s
        del self._vtime_5s
        del self._ld_speed_3min
        del self._vdist_3min
        del self._vtime_3min
        del self._timestamp_5s
        del self._timestamp_3min

    def get_sessions_in_split(self) -> List[Path]:
        """Return a list of paths that contains the simulation sessions in the split"""
        if not os.path.exists(self.session_splits):
            print("No train_test_split.json file found, please use `preprocess/simbarca/choose_train_test.py`")
            return []
            
        with open(self.session_splits, "r") as f:
            session_ids = json.load(f)[self.split]

        sessions_in_split = [os.path.join(self.data_root, self.session_folder_pattern.replace("*", "{:03d}".format(x))) for x in sorted(session_ids)]
        
        return sessions_in_split


    def get_session_properties(self):
        
        def session_number_from_path(path):
            import re
            return int(re.search(r"session_(\d+)", str(path)).group(1))
        
        sessions_in_split = self.get_sessions_in_split()
            
        session_ids = []
        for f in sessions_in_split:
            #scale = json.load(open(os.path.join(f, "settings.json"), 'r'))["global_scale"]
            session_id = session_number_from_path(f)
            session_ids.append(session_id)
            #demand_scales.append(scale)
        
        return {
            "session_ids": session_ids,
        }


    def init_graph_structure(self):
        """ read the graph structure for the road network from Aimsun-exported metadata.
        """
        
        connections = pd.read_csv(os.path.join(self.metadata_folder, "connections.csv"),
            dtype={
                "turn": int,
                "org": int,
                "dst": int,
                "intersection": int,
                "length": float,
            },
        )
        link_bboxes = pd.read_csv(os.path.join(self.metadata_folder, "link_bboxes_clustered.csv"),
            dtype={
                "id": int,
                "from_x": float,
                "from_y": float,
                "to_x": float,
                "to_y": float,
                "length": float,
                "out_ang": float,
                "num_lanes": int,
            },
        )
        with open(os.path.join(self.metadata_folder, "intersec_polygon.json")) as f:
            intersection_polygon = json.load(f)

        link_bboxes = link_bboxes.sort_values(by=["id"])
        section_ids_sorted = link_bboxes["id"].to_numpy()
        section_id_to_index = {link_id.item(): index for index, link_id in enumerate(section_ids_sorted)}
        index_to_section_id = {index: section_id.item() for index, section_id in enumerate(section_ids_sorted)}
        node_coordinates = link_bboxes[["c_x", "c_y"]].to_numpy()
        segment_lengths = link_bboxes["length"].to_numpy()
        num_lanes = link_bboxes["num_lanes"].to_numpy()
        
        adjacency_matrix = np.zeros((len(section_ids_sorted), len(section_ids_sorted)))
        weighted_adjacency_matrix = np.zeros((len(section_ids_sorted), len(section_ids_sorted)))
        for row in connections.itertuples():
            adjacency_matrix[section_id_to_index[row.org], section_id_to_index[row.dst]] = 1
            link_weight = 2/(segment_lengths[section_id_to_index[row.org]] + segment_lengths[section_id_to_index[row.dst]])
            weighted_adjacency_matrix[section_id_to_index[row.org], section_id_to_index[row.dst]] = link_weight
            # make it symmetric 
            adjacency_matrix[section_id_to_index[row.dst], section_id_to_index[row.org]] = 1
            weighted_adjacency_matrix[section_id_to_index[row.dst], section_id_to_index[row.org]] = link_weight
        edge_index = np.array(adjacency_matrix.nonzero())
        
        self.adjacency = adjacency_matrix
        self.weighted_adjacency = weighted_adjacency_matrix
        self.segment_lengths = segment_lengths
        self.edge_index = edge_index
        self.node_coordinates = node_coordinates
        self.section_ids_sorted = section_ids_sorted
        self.index_to_section_id = index_to_section_id
        self.section_id_to_index = section_id_to_index
        self.intersection_polygon = intersection_polygon
        self.num_lanes = num_lanes