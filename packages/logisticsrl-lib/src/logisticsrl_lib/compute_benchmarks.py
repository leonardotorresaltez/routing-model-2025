import os
import random

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# import wandb
# from configs.config import parse_args
# from core.envs.tsp_env import TSPEnv
# from core.models.agent import REINFORCEAgent
# from core.utils.data_loader import MDVRPDataLoader
# from core.utils.evaluation_utils import evaluate_solution
# from core.utils.visualization_utils_plotly import create_routing_graph, visualize_routing_solution
from logisticsrl_lib.configs.config import parse_args
from logisticsrl_lib.reinforcelearning.tsp_env import TSPEnv
from logisticsrl_lib.reinforcelearning.agent import REINFORCEAgent
from loader_lib.data_loader import FleetStatus, MDVRPDataLoader, TruckState
from common_lib.evaluation_utils import evaluate_solution
from common_lib.visualization_utils_plotly import create_routing_graph, visualize_routing_solution
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs("checkpoints", exist_ok=True)
    
    loader = MDVRPDataLoader(data_dir=cfg.data_dir)
    data = loader.load_data()
    node_to_idx = data["node_to_idx"]
    idx_to_node = data["idx_to_node"]

    # # Update config based on loaded data
    cfg.num_nodes = data["num_nodes"] 
        
        
    truck_starts = [truck.depot_idx for truck in data["trucks"]]
    print(f"Truck start positions (indices): {truck_starts}")

    time_matrix = data["time_matrix"]
    unique_truck_starts = list(set(truck_starts))
    print(f"Unique truck start positions: {unique_truck_starts}")

    time_matrix_from_truck_starts = time_matrix[unique_truck_starts, :]
    time_matrix_to_truck_starts = time_matrix[:, unique_truck_starts].T

    go_and_return_time_matrix = time_matrix_from_truck_starts + time_matrix_to_truck_starts
    # print("Go and return time matrix:\n", go_and_return_time_matrix)

    time_wrt_closest_truck_start = go_and_return_time_matrix.min(dim=0).values
    # print("time_wrt_closest_truck_start: ", time_wrt_closest_truck_start)

    unfeasible_nodes = (time_wrt_closest_truck_start > cfg.max_daily_delivery_time_each_truck).nonzero(as_tuple=True)[0].tolist()
    print(f"Unfeasible nodes (time to closest truck start > {cfg.max_daily_delivery_time_each_truck}): {unfeasible_nodes}")

    # print("CHECKKKKKK")
    # print(data.keys())
    customers_df = data["customers_df"]
    customers_df["idx"] = customers_df["id_customer"].map(node_to_idx)
    customers_df = customers_df.set_index("idx")
    customers_df["feasible"] = True
    customers_df.loc[unfeasible_nodes, "feasible"] = False

    customers = data["customers"]
    depots = data["depots"]
    


    # CLUSTERING:
    # 1. Filter for feasible customers and extract coordinates
    feasible_mask = customers_df['feasible'] == True
    feasible_coords = customers_df.loc[feasible_mask, ['latitude', 'longitude']]

    # 2. Define number of clusters based on truck_starts
    n_clusters = len(truck_starts)

    # 3. Perform K-means clustering
    # Note: n_init='auto' or 10 is recommended depending on your sklearn version
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(feasible_coords)

    # 4. Assign results back to the original DataFrame
    # We initialize the column with -1 (or any sentinel value) for non-feasible nodes
    customers_df['cluster'] = -1
    customers_df.loc[feasible_mask, 'cluster'] = clusters
    # print(customers_df)
    cluster_counts = customers_df['cluster'].value_counts().sort_index()
    print("cluster_counts", cluster_counts)

    

    customers_df['cluster'] = customers_df['cluster'].astype('category')
    ax = customers_df.plot.scatter(
        x='longitude', 
        y='latitude', 
        c=customers_df['cluster'].cat.codes, # Use category codes for color mapping
        cmap='tab10',                        # A discrete colormap is better for categories
        colorbar=False,                      # Disable the continuous colorbar
        figsize=(10, 6)
    )

    plt.title("Clusters as Categorical Groups")
    plt.savefig("checkpoints/clusters_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()

    def solve_greedy_route(cluster_nodes, depot_idx, time_matrix, max_time):
        """
        Args:
            cluster_nodes (list): Indices of customers in this cluster (must match time_matrix indices).
            depot_idx (int): Index of the depot in time_matrix.
            time_matrix (Tensor): NxN matrix of travel times.
            max_time (float): Maximum allowed time (including return to depot).
        
        Returns:
            route (list): Ordered list of visited node indices.
            total_time (float): Total duration of the route.
        """
        current_node = depot_idx
        route = [depot_idx]
        current_time = 0.0
        
        # Keep track of unvisited nodes in this cluster
        unvisited = set(cluster_nodes)
        if depot_idx in unvisited:
            unvisited.remove(depot_idx)
            
        while unvisited:
            # 1. Calculate times from current_node to all unvisited nodes
            # We convert set to list for indexing
            candidates = list(unvisited)
            
            # Get travel times from tensor
            # time_matrix[current_node, candidates] gives shape (len(candidates),)
            travel_times = time_matrix[current_node, candidates]
            
            # 2. Find closest candidate (greedy step)
            min_travel_time, min_idx = torch.min(travel_times, dim=0)
            closest_candidate = candidates[min_idx.item()]
            travel_time = min_travel_time.item()
            
            # 3. Feasibility Check: Can we go there AND return to depot?
            time_to_return = time_matrix[closest_candidate, depot_idx].item()
            
            if current_time + travel_time + time_to_return <= max_time:
                # Move to node
                current_node = closest_candidate
                current_time += travel_time
                route.append(current_node)
                unvisited.remove(current_node)
            else:
                # If we can't visit the closest one, we likely can't visit others efficiently 
                # (simple greedy limitation). We stop here.
                break
                
        # 4. Return to Depot
        return_time = time_matrix[current_node, depot_idx].item()
        current_time += return_time
        route.append(depot_idx)
        
        return route, current_time

    results = {}

    # Iterate over each cluster (excluding the -1 unfeasible group)
    for cluster_id in range(n_clusters):
        if cluster_id == -1: 
            continue
            
        # Get the global indices of customers in this cluster
        # NOTE: This assumes the DataFrame index aligns with time_matrix indices.
        # If 'idx' is a column, use customers_df[...]['idx'].tolist()
        cluster_customer_indices = customers_df[customers_df['cluster'] == cluster_id].index.tolist()
        
        depot_idx = truck_starts[cluster_id] # ALEX: FIX: iF THERE IS MORE THAN ONE DEPOT, ASSIGN TRUCKS TO THEIR CLOSEST CLUSTER
        
        # Run Greedy
        route, total_time = solve_greedy_route(
            cluster_nodes=cluster_customer_indices,
            depot_idx=depot_idx,
            time_matrix=time_matrix,
            max_time=cfg.max_daily_delivery_time_each_truck
        )
        
        results[cluster_id] = {
            "route": route,
            "time": total_time,
            "count": len(route) - 2 # Subtract start/end depot
        }
        
        print(f"Cluster {cluster_id}: Visited {len(route)-2} out of {len(cluster_customer_indices)} customers. Time: {total_time:.2f}/{cfg.max_daily_delivery_time_each_truck}")
        print(f"Route: {route}\n")
    
    tours = [results[cluster_id]["route"][:-1] for cluster_id in range(n_clusters) if cluster_id in results]
    total_destinations_visited, total_time, pct_intersections = evaluate_solution(tours, data, truck_starts, cfg)
    print(f"Total destinations visited: {total_destinations_visited}, Total time: {total_time:.2f}, Percentage of intersections: {pct_intersections:.2%}")

    for customer in customers:
        if customer.idx in [node for tour in tours for node in tour]:
            customer.delivered = True
        else:
            customer.delivered = False
    
    G = create_routing_graph(depots, customers, tours, truck_starts)
    visualize_routing_solution(G, step=0, title_suffix="", save_path=f"checkpoints/viz_kmeans_and_greedy_{cfg.data_dir}.html")

if __name__ == "__main__":
    train()