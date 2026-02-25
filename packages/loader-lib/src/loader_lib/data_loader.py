import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

@dataclass
class TruckState:
    total_time: float = 0.0
    tour: list = field(default_factory=list)
    position: int = None  # current node index
    noop: bool = False  # Flag to indicate if the last action was a NO-OP

@dataclass
class FleetStatus:
    active_truck: int = 0
    trucklist: dict[int, TruckState] = field(default_factory=dict)  # key: truck_id, value: TruckState(total_time, tour)
    truck_starts: list[int] = field(default_factory=list)  # List of starting depot indices for each truck
    source_mask: np.ndarray = None  # Mask to identify source nodes (depots)
    time_matrix: dict = None  # Time matrix for travel times between nodes
    nodes: torch.Tensor = None  # Node features (e.g., coordinates, time profiles)
    
    def truck_positions(self):
        return np.array([state.position for state in self.trucklist.values()], dtype=np.int64)
    
    def num_nodes(self):
        return self.nodes.shape[0] if self.nodes is not None else 0    
    
    def all_tours(self):
        return [state.tour for state in self.trucklist.values()]
    def num_trucks(self):
        return len(self.truck_starts)

@dataclass
class Node:
    """Base class for a graph location (depot or customer)."""
    id_str: str
    idx: int
    lat: float
    lon: float
    isSource: bool = field(init=False)
    
    def location(self) -> Tuple[float, float]:
        return (self.lat, self.lon)

@dataclass
class Customer(Node):
    road_access_type: str
    delivered: bool = False
    
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
class Depot(Node):
    truck_fleet: List[int] = field(default_factory=list)

class MDVRPDataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Pointing to the project root/data/data_version_2
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "data_version_2"
        else:
            # Si es ruta absoluta, úsala directamente; si es relativa, únete a la raíz del proyecto
            data_dir_path = Path(data_dir)
            if data_dir_path.is_absolute():
                self.data_dir = data_dir_path
            else:
                self.data_dir = Path(__file__).resolve().parent.parent.parent / ".." / ".." / "data" / data_dir
        self.node_to_idx = {}
        self.idx_to_node = {}

    def load_data(self) -> Dict:
        """
        Loads all CSVs and returns a consolidated dictionary of objects and tensors.
        """
        # Step 1. Load DataFrames
        depot_df = pd.read_csv(os.path.join(self.data_dir, "selected_depot.csv"))
        customer_df = pd.read_csv(os.path.join(self.data_dir, "selected_customers.csv"))
        truck_df = pd.read_csv(os.path.join(self.data_dir, "selected_trucks.csv"))

        # Step 2. Map IDs to Indices
        # Keep depots first so we can label nodes accordingly
        all_node_ids = list(depot_df["id_depot"]) + list(customer_df["id_customer"])
        self.node_to_idx = {node_id: i for i, node_id in enumerate(all_node_ids)}
        self.idx_to_node = {i: node_id for node_id, i in self.node_to_idx.items()}

        # Number of nodes (depots + customers) -- used to initialize node containers
        num_nodes = len(all_node_ids)

        # Step 3. Initialize Objects
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
            
        nodes: List[Node] = [None] * num_nodes
        for d in depots:
            d.isSource = True
            nodes[d.idx] = d
        for c in customers:
            c.isSource = False
            nodes[c.idx] = c            

        # Step 4. Build Travel Time Matrix
        time_matrix = np.zeros((num_nodes, num_nodes))
        
        time_files = glob.glob(os.path.join(self.data_dir, "time_between_nodes_*.csv"))
        for f in time_files:
            df_chunk = pd.read_csv(f)
            for _, r in df_chunk.iterrows():
                id1, id2 = r["id_node1"], r["id_node2"]
                if id1 in self.node_to_idx and id2 in self.node_to_idx:
                    i, j = self.node_to_idx[id1], self.node_to_idx[id2]
                    time_matrix[i, j] = time_matrix[j, i] = r["time_h"]

        # Step 5. Feature Engineering (Using time proximity profiles)
        time_tensor = torch.tensor(time_matrix, dtype=torch.float32)
        # Normalize features by max time for neural network stability
        node_features = time_tensor / (time_tensor.max() + 1e-9)



        # print(time_matrix)
        # print(time_tensor)
        # print(node_features)
        return {
            "node_features": node_features,
            "time_matrix": time_tensor,
            "depots": depots,
            "customers": customers,
            "customers_df": customer_df,
            "nodes": nodes,
            "trucks": trucks,
            "num_nodes": num_nodes,
            "node_to_idx": self.node_to_idx,
            "idx_to_node": self.idx_to_node
        }

