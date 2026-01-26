import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class Customer:
    id_str: str
    idx: int
    lat: float
    lon: float
    road_access_type: str
    
@dataclass
class Truck:
    id: int
    depot_id_str: str
    depot_idx: int
    max_weight: float
    height: float
    length: float
    width: float
    volume: float = field(init=False)

    def __post_init__(self):
        self.volume = self.height * self.length * self.width

@dataclass
class Depot:
    id_str: str
    idx: int
    lat: float
    lon: float
    truck_fleet: List[int] = field(default_factory=list)

class MDVRPDataLoader:
    def __init__(self, data_dir=r"c:\Users\clara.gregori\projects\VisualStudioCode\routing-model-2025\data\data_version_2"):
        self.data_dir = data_dir
        self.node_to_idx = {}
        self.idx_to_node = {}

    def load_data(self) -> Dict:
        """
        Loads all CSVs and returns a consolidated dictionary of objects and tensors.
        """
        # 1. Load DataFrames
        depot_df = pd.read_csv(os.path.join(self.data_dir, "selected_depot.csv"))
        customer_df = pd.read_csv(os.path.join(self.data_dir, "selected_customers.csv"))
        truck_df = pd.read_csv(os.path.join(self.data_dir, "selected_trucks.csv"))

        # 2. Map IDs to Indices
        all_node_ids = list(depot_df["id_depot"]) + list(customer_df["id_customer"])
        self.node_to_idx = {node_id: i for i, node_id in enumerate(all_node_ids)}
        self.idx_to_node = {i: node_id for node_id, i in self.node_to_idx.items()}

        # 3. Initialize Objects
        depots = []
        for _, r in depot_df.iterrows():
            idx = self.node_to_idx[r["id_depot"]]
            depots.append(Depot(r["id_depot"], idx, r["latitude"], r["longitude"]))

        customers = []
        for _, r in customer_df.iterrows():
            idx = self.node_to_idx[r["id_customer"]]
            customers.append(Customer(r["id_customer"], idx, r["latitude"], r["longitude"], r["vehicle_access_type"]))

        trucks = []
        for _, r in truck_df.iterrows():
            d_idx = self.node_to_idx[r["id_depot"]]
            t = Truck(r["id_truck"], r["id_depot"], d_idx, r["max_weight"], r["height"], r["length"], r["width"])
            trucks.append(t)
            depots[d_idx].truck_fleet.append(t.id)

        # 4. Build Travel Time Matrix
        num_nodes = len(all_node_ids)
        time_matrix = np.zeros((num_nodes, num_nodes))
        
        time_files = glob.glob(os.path.join(self.data_dir, "time_between_nodes_*.csv"))
        for f in time_files:
            df_chunk = pd.read_csv(f)
            for _, r in df_chunk.iterrows():
                id1, id2 = r["id_node1"], r["id_node2"]
                if id1 in self.node_to_idx and id2 in self.node_to_idx:
                    i, j = self.node_to_idx[id1], self.node_to_idx[id2]
                    time_matrix[i, j] = time_matrix[j, i] = r["time_h"]

        # 5. Feature Engineering (Using time proximity profiles)
        time_tensor = torch.tensor(time_matrix, dtype=torch.float32)
        # Normalize features by max time for neural network stability
        node_features = time_tensor / (time_tensor.max() + 1e-9)

        return {
            "node_features": node_features,
            "time_matrix": time_tensor,
            "depots": depots,
            "customers": customers,
            "trucks": trucks,
            "num_nodes": num_nodes,
            "node_to_idx": self.node_to_idx,
            "idx_to_node": self.idx_to_node
        }

