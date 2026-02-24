import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

@dataclass
class Node:
    id_str: str
    idx: int
    lat: float
    lon: float
    isSource: bool = False
    cluster_id: int = -1
    def location(self) -> Tuple[float, float]:
        return (self.lat, self.lon)

@dataclass
class Customer(Node):
    road_access_type: str = ""
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
    target_cluster: int = -1 
    volume: float = field(init=False)

    def __post_init__(self):
        self.volume = self.height * self.length * self.width

@dataclass
class Depot(Node):
    truck_fleet: List[int] = field(default_factory=list)
    isSource: bool = True

class MDVRPDataLoader:
    def __init__(self, config_data_dir: str):
        # Dynamically find data path based on Config
        base_path = Path(__file__).resolve().parent.parent.parent / "data"
        self.data_dir = base_path / config_data_dir
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        self.node_to_idx = {}
        self.idx_to_node = {}

    def load_data(self) -> Dict:
        print(f"--- Loading data from: {self.data_dir} ---")
        depot_df = pd.read_csv(self.data_dir / "selected_depot.csv")
        customer_df = pd.read_csv(self.data_dir / "selected_customers.csv")
        truck_df = pd.read_csv(self.data_dir / "selected_trucks.csv")

        # 1. K-Means Clustering (Dynamic based on fleet size)
        num_clusters = len(truck_df)
        cust_coords = customer_df[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        customer_df['cluster_id'] = kmeans.fit_predict(cust_coords)
        cluster_centers = kmeans.cluster_centers_

        # 2. Node Mapping
        all_ids = list(depot_df["id_depot"]) + list(customer_df["id_customer"])
        self.node_to_idx = {nid: i for i, nid in enumerate(all_ids)}
        self.idx_to_node = {i: nid for nid, i in self.node_to_idx.items()}
        num_nodes = len(all_ids)

        # 3. Object Creation
        depots = [Depot(r["id_depot"], self.node_to_idx[r["id_depot"]], r["latitude"], r["longitude"]) for _, r in depot_df.iterrows()]
        customers = [Customer(r["id_customer"], self.node_to_idx[r["id_customer"]], r["latitude"], r["longitude"], r["vehicle_access_type"]) for _, r in customer_df.iterrows()]
        for c, cluster_id in zip(customers, customer_df['cluster_id']): 
            c.cluster_id = int(cluster_id)

        trucks = []
        for _, r in truck_df.iterrows():
            t = Truck(r["id_truck"], r["id_depot"], self.node_to_idx[r["id_depot"]], r["max_weight"], r["height"], r["length"], r["width"])
            depot_loc = depot_df[depot_df["id_depot"] == r["id_depot"]][['latitude', 'longitude']].values
            # Logic: Assign truck to the closest cluster of customers
            t.target_cluster = int(np.argmin(np.linalg.norm(cluster_centers - depot_loc, axis=1)))
            trucks.append(t)

        nodes = [None] * num_nodes
        for d in depots: nodes[d.idx] = d
        for c in customers: nodes[c.idx] = c

        coords = np.array([[n.lat, n.lon] for n in nodes])
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        demands = torch.tensor([0.0 if n.isSource else 1.0 for n in nodes]).unsqueeze(1)
        visited = torch.zeros((num_nodes, 1))
        
        # NEW: Normalize Cluster IDs to [0, 1] range
        c_ids = torch.tensor([n.cluster_id for n in nodes], dtype=torch.float32).unsqueeze(1)
       
        
        # Combined features [Lat, Lon, Demand, Visited, ClusterID]
        node_features = torch.cat([
            coords_tensor, 
            demands, 
            visited, 
            c_ids
        ], dim=1)
        # 5. Time Matrix Loading
        time_matrix = torch.zeros((num_nodes, num_nodes))
        time_files = glob.glob(str(self.data_dir / "time_between_nodes_*.csv"))
        for f in time_files:
            df = pd.read_csv(f)
            for _, r in df.iterrows():
                if r["id_node1"] in self.node_to_idx and r["id_node2"] in self.node_to_idx:
                    i, j = self.node_to_idx[r["id_node1"]], self.node_to_idx[r["id_node2"]]
                    time_matrix[i, j] = time_matrix[j, i] = float(r["time_h"])

        return {
            "node_features": node_features,
            "cluster_ids": c_ids,
            "time_matrix": time_matrix,
            "trucks": trucks,
            "num_nodes": num_nodes,
            "depot_indices": [d.idx for d in depots],
            "customers": customers,
            "depots": depots,
            "nodes" : nodes,
            "node_to_idx":self.node_to_idx,
            "idx_to_node":self.idx_to_node
        }