"""
Utility functions for simplified fleet routing.

This module provides helper functions for:
- Distance calculations
- Feasibility checking
- Graph construction with NetworkX
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from env.types_simple import Customer, Truck, Depot, TruckState


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        p1: (x, y) tuple for point 1
        p2: (x, y) tuple for point 2
    
    Returns:
        Euclidean distance
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def can_truck_serve_customer(
    truck_state: TruckState,
    customer: Customer
) -> bool:
    """
    Check if truck has capacity to serve customer.
    
    Args:
        truck_state: current truck state
        customer: customer to potentially serve
    
    Returns:
        True if truck has enough capacity, False otherwise
    """
    return truck_state.remaining_capacity() >= customer.weight


def build_distance_matrix(
    customers: List[Customer],
    depots: List[Depot],
    trucks: List[Truck]
) -> np.ndarray:
    """
    Build distance matrix for all locations.
    
    Matrix structure: [depots, customers]
    - Rows 0 to D-1: depots
    - Rows D to D+N-1: customers
    
    Args:
        customers: list of Customer objects
        depots: list of Depot objects
        trucks: list of Truck objects (used for count only)
    
    Returns:
        (D + N) x (D + N) distance matrix where distance[i, j] is
        Euclidean distance from location i to location j
    """
    num_depots = len(depots)
    num_customers = len(customers)
    total_locations = num_depots + num_customers
    
    distance_matrix = np.zeros((total_locations, total_locations))
    
    # All locations: depots first, then customers
    locations = [d.location() for d in depots] + [c.location() for c in customers]
    
    for i in range(total_locations):
        for j in range(total_locations):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    
    return distance_matrix


def build_routing_graph(
    customers: List[Customer],
    depots: List[Depot],
    trucks: List[Truck],
    truck_states: List[TruckState],
    feasibility_masks: np.ndarray
) -> nx.MultiDiGraph:
    """
    Build a NetworkX directed graph representing the routing problem state.
    
    Graph structure:
    - Nodes: depot_0, depot_1, ..., customer_0, customer_1, ...
    - Edges: from each truck's current location to:
      - All unvisited customers (if feasible)
      - All depots (always feasible)
    
    Args:
        customers: list of Customer objects
        depots: list of Depot objects
        trucks: list of Truck objects
        truck_states: current state of each truck
        feasibility_masks: (T, N+1) boolean array where
                          feasibility_masks[t, c] = 1 if truck t can serve customer c
    
    Returns:
        NetworkX directed graph with truck and customer/depot nodes
    """
    G = nx.MultiDiGraph()
    
    num_depots = len(depots)
    num_customers = len(customers)
    num_trucks = len(trucks)
    
    # Add depot nodes
    for depot in depots:
        G.add_node(f"depot_{depot.id}", type="depot", x=depot.x, y=depot.y)
    
    # Add customer nodes
    for customer in customers:
        G.add_node(f"customer_{customer.id}", type="customer", 
                   x=customer.x, y=customer.y, weight=customer.weight)
    
    # Add truck nodes
    for i, truck_state in enumerate(truck_states):
        pos = truck_state.current_location
        G.add_node(f"truck_{i}", type="truck", 
                   x=pos[0], y=pos[1], load=truck_state.current_load)
    
    # Add edges from each truck to feasible customers and depots
    for truck_id, truck_state in enumerate(truck_states):
        truck_pos = truck_state.current_location
        
        # Edges to unvisited customers (if feasible)
        for customer_id in range(num_customers):
            if customer_id not in truck_state.visited_customers:
                # Check feasibility from mask
                if feasibility_masks[truck_id, customer_id] == 1:
                    customer = customers[customer_id]
                    distance = euclidean_distance(truck_pos, customer.location())
                    
                    G.add_edge(f"truck_{truck_id}", f"customer_{customer_id}",
                              distance=distance, weight=customer.weight)
        
        # Edges to all depots (always feasible - can return home anytime)
        for depot in depots:
            distance = euclidean_distance(truck_pos, depot.location())
            G.add_edge(f"truck_{truck_id}", f"depot_{depot.id}",
                      distance=distance, is_depot_return=True)
    
    return G


def compute_utilization_reward(truck_states: List[TruckState]) -> float:
    """
    Compute utilization reward based on how full trucks are.
    
    Reward structure:
    - utilization > 0.8: +5.0 (strong bonus for full trucks)
    - 0.5 < utilization <= 0.8: +2.0 (moderate bonus)
    - utilization <= 0.3: -1.0 (penalty for nearly empty trucks)
    
    Args:
        truck_states: current state of all trucks
    
    Returns:
        Total utilization reward across all trucks
    """
    reward = 0.0
    
    for truck_state in truck_states:
        utilization = truck_state.utilization()
        
        if utilization > 0.8:
            reward += 5.0
        elif utilization > 0.5:
            reward += 2.0
        elif utilization < 0.3:
            reward -= 1.0
    
    return reward
