"""
Convert flat observations to graph representations for Graph Pointer Network.

This module extracts node features and adjacency matrices from flat observation
vectors, enabling the GraphPointerNetwork to process fleet routing problems as
explicit graph structures.
"""

import numpy as np
import torch
import networkx as nx


def reconstruct_observation_components(
    observation: np.ndarray, 
    num_trucks: int, 
    num_customers: int
) -> dict:
    """
    Reconstruct individual components from flat observation vector.
    
    Reverses the flattening process in sb3_wrapper.py to extract meaningful
    components from the concatenated observation vector.
    
    Args:
        observation: Flat 1D array of size 5*T + 4*N
        num_trucks: Number of trucks (T)
        num_customers: Number of customers (N)
    
    Returns:
        Dictionary with keys:
            - truck_positions: [T, 2] array of (x, y) coordinates
            - truck_loads: [T] array of current loads
            - truck_capacities: [T] array of max capacities
            - truck_utilization: [T] array of load/capacity ratios
            - customer_locations: [N, 2] array of (x, y) coordinates
            - customer_weights: [N] array of delivery demands
            - unvisited_mask: [N] array of visit status (1=unvisited, 0=visited)
    """
    T = num_trucks
    N = num_customers
    
    idx = 0
    
    truck_positions = observation[idx:idx+2*T].reshape(T, 2)
    idx += 2*T
    
    truck_loads = observation[idx:idx+T]
    idx += T
    
    truck_capacities = observation[idx:idx+T]
    idx += T
    
    truck_utilization = observation[idx:idx+T]
    idx += T
    
    customer_locations = observation[idx:idx+2*N].reshape(N, 2)
    idx += 2*N
    
    customer_weights = observation[idx:idx+N]
    idx += N
    
    unvisited_mask = observation[idx:idx+N]
    
    return {
        'truck_positions': truck_positions,
        'truck_loads': truck_loads,
        'truck_capacities': truck_capacities,
        'truck_utilization': truck_utilization,
        'customer_locations': customer_locations,
        'customer_weights': customer_weights,
        'unvisited_mask': unvisited_mask,
    }


def build_node_features(
    truck_positions: np.ndarray,
    customer_locations: np.ndarray
) -> torch.Tensor:
    """
    Build node feature matrix combining truck and customer positions.
    
    Concatenates truck and customer positions to create a unified node feature
    matrix where each row represents either a truck or customer node.
    
    Args:
        truck_positions: [T, 2] array of truck coordinates
        customer_locations: [N, 2] array of customer coordinates
    
    Returns:
        Tensor of shape [T+N, 2] with all node positions
    """
    all_positions = np.vstack([truck_positions, customer_locations])
    return torch.FloatTensor(all_positions)


def build_adjacency_matrix(
    node_features: torch.Tensor,
    k_neighbors: int = 5,
    threshold: float = None
) -> torch.Tensor:
    """
    Build adjacency matrix based on spatial proximity (k-nearest neighbors).
    
    Creates a sparse adjacency matrix where nodes are connected to their
    k nearest neighbors by Euclidean distance. This captures the spatial
    structure of the routing problem.
    
    Args:
        node_features: [N_nodes, 2] tensor of node positions
        k_neighbors: Number of nearest neighbors to connect (default: 5)
        threshold: Alternative to k_neighbors - connect if distance < threshold
                  If specified, k_neighbors is ignored
    
    Returns:
        Adjacency matrix of shape [N_nodes, N_nodes] with values {0, 1}
    """
    n_nodes = node_features.shape[0]
    
    distances = torch.cdist(node_features, node_features)
    adjacency = torch.zeros(n_nodes, n_nodes)
    
    if threshold is not None:
        adjacency = (distances < threshold).float()
    else:
        for i in range(n_nodes):
            sorted_distances, indices = torch.topk(
                distances[i], 
                k=min(k_neighbors + 1, n_nodes),
                largest=False
            )
            adjacency[i, indices] = 1.0
    
    symmetrize = (adjacency + adjacency.T) > 0
    adjacency = symmetrize.float()
    
    return adjacency


def observation_to_graph(
    observation: np.ndarray,
    num_trucks: int,
    num_customers: int,
    k_neighbors: int = 5
) -> tuple:
    """
    Convert flat observation to graph representation (node features, adjacency).
    
    This is the main conversion function that orchestrates the extraction and
    graph building process. It transforms the flat 80-dimensional observation
    into an explicit graph structure suitable for GraphPointerNetwork processing.
    
    Args:
        observation: Flat observation vector of size 5*T + 4*N
        num_trucks: Number of trucks
        num_customers: Number of customers
        k_neighbors: Number of nearest neighbors for adjacency (default: 5)
    
    Returns:
        Tuple of (node_features, adjacency_matrix):
            - node_features: [T+N, 2] tensor of node positions
            - adjacency_matrix: [T+N, T+N] tensor representing graph structure
    """
    components = reconstruct_observation_components(
        observation, num_trucks, num_customers
    )
    
    node_features = build_node_features(
        components['truck_positions'],
        components['customer_locations']
    )
    
    adjacency_matrix = build_adjacency_matrix(
        node_features,
        k_neighbors=k_neighbors
    )
    
    return node_features, adjacency_matrix


def create_networkx_graph(
    node_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    node_attributes: dict = None
) -> nx.Graph:
    """
    Create NetworkX graph from node features and adjacency matrix.
    
    Useful for visualization and analysis of the routing graph structure.
    Can optionally attach node attributes like loads, capacities, demands.
    
    Args:
        node_features: [N_nodes, 2] tensor of node positions
        adjacency_matrix: [N_nodes, N_nodes] adjacency tensor
        node_attributes: Optional dict with node-level attributes
                        Keys should be attribute names, values should be lists
    
    Returns:
        NetworkX Graph object with nodes, edges, and optional attributes
    """
    n_nodes = node_features.shape[0]
    G = nx.Graph()
    
    for i in range(n_nodes):
        x, y = float(node_features[i, 0]), float(node_features[i, 1])
        G.add_node(i, pos=(x, y))
        
        if node_attributes:
            for attr_name, attr_values in node_attributes.items():
                if i < len(attr_values):
                    G.nodes[i][attr_name] = float(attr_values[i])
    
    adj_np = adjacency_matrix.cpu().numpy() if isinstance(adjacency_matrix, torch.Tensor) else adjacency_matrix
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_np[i, j] > 0:
                G.add_edge(i, j)
    
    return G
