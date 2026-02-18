import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans   # <-- ADDED

@dataclass
class TruckState:
    total_time: float = 0.0
    tour: list = field(default_factory=list)
    position: int = None

@dataclass
class FleetStatus:
    active_truck: int = 0
    trucklist: dict[int, TruckState] = field(default_factory=dict)
    truck_starts: list[int] = field(default_factory=list)
    source_mask: np.ndarray = None
    time_matrix: dict = None
    nodes: torch.Tensor = None
    
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
    delivered: bool = False  # <-- ADDED

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
    cluster_id: int = 0 

class MDVRPDataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "data_version_1"
        else:
            data_dir_path = Path(data_dir)
            if data_dir_path.is_absolute():
                self.data_dir = data_dir_path
            else:
                self.data_dir = Path(__file__).resolve().parent.parent.parent / ".." / ".." / "data" / data_dir

        self.node_to_idx = {}
        self.idx_to_node = {}

    def load_data(self) -> Dict:

        depot_df = pd.read_csv(os.path.join(self.data_dir, "selected_depot.csv"))
        customer_df = pd.read_csv(os.path.join(self.data_dir, "selected_customers.csv"))
        truck_df = pd.read_csv(os.path.join(self.data_dir, "selected_trucks.csv"))

  
        num_clusters = len(truck_df) 
        cust_coords = customer_df[['latitude', 'longitude']].values

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        customer_df['cluster_id'] = kmeans.fit_predict(cust_coords)


        all_node_ids = list(depot_df["id_depot"]) + list(customer_df["id_customer"])
        self.node_to_idx = {node_id: i for i, node_id in enumerate(all_node_ids)}
        self.idx_to_node = {i: node_id for node_id, i in self.node_to_idx.items()}

        num_nodes = len(all_node_ids)


        depots = []
        for _, r in depot_df.iterrows():
            idx = self.node_to_idx[r["id_depot"]]
            d = Depot(r["id_depot"], idx, r["latitude"], r["longitude"])
            d.cluster_id = 0  # depots belong to cluster 0
            depots.append(d)

        customers = []
        for _, r in customer_df.iterrows():
            idx = self.node_to_idx[r["id_customer"]]
            c = Customer(
                r["id_customer"],
                idx,
                r["latitude"],
                r["longitude"],
                r["vehicle_access_type"],
            )
            c.cluster_id = int(r["cluster_id"]) 
            customers.append(c)

      
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

        time_matrix = np.zeros((num_nodes, num_nodes))
        time_files = glob.glob(os.path.join(self.data_dir, "time_between_nodes_*.csv"))

        for f in time_files:
            df_chunk = pd.read_csv(f)
            for _, r in df_chunk.iterrows():
                id1, id2 = r["id_node1"], r["id_node2"]
                if id1 in self.node_to_idx and id2 in self.node_to_idx:
                    i, j = self.node_to_idx[id1], self.node_to_idx[id2]
                    time_matrix[i, j] = time_matrix[j, i] = r["time_h"]

      
        coords = torch.tensor([[n.lat, n.lon] for n in nodes], dtype=torch.float32)
        demands = torch.tensor([0.0 if n.isSource else 1.0 for n in nodes], dtype=torch.float32).unsqueeze(1)
        visited = torch.zeros((num_nodes, 1), dtype=torch.float32)

        cluster_ids = torch.tensor(
            [n.cluster_id for n in nodes],
            dtype=torch.float32
        ).unsqueeze(1)

        node_features = torch.cat([coords, demands, visited, cluster_ids], dim=1)
        node_to_cluster = {n.idx: n.cluster_id for n in nodes}
       
        return {
            "node_features": node_features,
            "coords": coords,
            "demands": demands,
            "cluster_ids": cluster_ids,
            "time_matrix": torch.tensor(time_matrix, dtype=torch.float32),
            "depots": depots,
            "customers": customers,
            "nodes": nodes,
            "trucks": trucks,
            "num_nodes": num_nodes,
            "node_to_cluster":node_to_cluster,
            "node_to_idx": self.node_to_idx,
            "idx_to_node": self.idx_to_node,
            "kmeans": kmeans,
        }