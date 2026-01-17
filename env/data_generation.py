"""
Data generation for simplified fleet routing problems.

Creates random problem instances with customers, trucks, and depots
for testing and training the routing environment.
"""

import numpy as np
from typing import List, Tuple
from env.types_simple import Customer, Truck, Depot


def generate_problem_instance(
    num_customers: int = 20,
    num_trucks: int = 5,
    num_depots: int = 2,
    area_size: float = 100.0,
    truck_capacity_range: Tuple[float, float] = (500.0, 1000.0),
    customer_weight_range: Tuple[float, float] = (10.0, 100.0),
    seed: int = None
) -> Tuple[List[Customer], List[Truck], List[Depot]]:
    """
    Generate a random fleet routing problem instance.
    
    Args:
        num_customers: number of customers to deliver to
        num_trucks: number of trucks in fleet
        num_depots: number of depots
        area_size: coordinates range is [0, area_size]
        truck_capacity_range: (min, max) capacity for trucks in kg
        customer_weight_range: (min, max) weight for customers in kg
        seed: random seed for reproducibility
    
    Returns:
        Tuple of (customers, trucks, depots)
    """
    if seed is not None:
        np.random.seed(seed)
    
    customers = []
    trucks = []
    depots = []
    
    for i in range(num_depots):
        depot = Depot(
            id=i,
            x=np.random.uniform(0, area_size),
            y=np.random.uniform(0, area_size)
        )
        depots.append(depot)
    
    for i in range(num_trucks):
        depot_id = i % num_depots
        capacity = np.random.uniform(*truck_capacity_range)
        truck = Truck(
            id=i,
            depot_id=depot_id,
            max_capacity=capacity
        )
        trucks.append(truck)
    
    for i in range(num_customers):
        customer = Customer(
            id=i,
            x=np.random.uniform(0, area_size),
            y=np.random.uniform(0, area_size),
            weight=np.random.uniform(*customer_weight_range)
        )
        customers.append(customer)
    
    return customers, trucks, depots


def generate_curriculum_instances(
    difficulty_levels: int = 5,
    base_customers: int = 10,
    base_trucks: int = 3,
    seed: int = None
) -> List[Tuple[List[Customer], List[Truck], List[Depot]]]:
    """
    Generate a curriculum of problem instances with increasing difficulty.
    
    Args:
        difficulty_levels: number of difficulty levels
        base_customers: customers in easiest problem
        base_trucks: trucks in easiest problem
        seed: random seed
    
    Returns:
        List of problem instances, ordered by difficulty
    """
    instances = []
    
    for level in range(difficulty_levels):
        scale_factor = 1.0 + (level * 0.5)
        num_customers = int(base_customers * scale_factor)
        num_trucks = max(2, int(base_trucks * (1 + level * 0.3)))
        
        customers, trucks, depots = generate_problem_instance(
            num_customers=num_customers,
            num_trucks=num_trucks,
            num_depots=2,
            area_size=100.0,
            seed=seed + level if seed is not None else None
        )
        
        instances.append((customers, trucks, depots))
    
    return instances


def calculate_optimal_bounds(
    customers: List[Customer],
    trucks: List[Truck],
    depots: List[Depot]
) -> Tuple[float, float]:
    """
    Calculate theoretical bounds on optimal solution.
    
    Args:
        customers: list of customers
        trucks: list of trucks
        depots: list of depots
    
    Returns:
        Tuple of (lower_bound, upper_bound) for minimum total distance
    
    Lower bound: sum of distances from each customer to nearest depot
    Upper bound: naive nearest-neighbor from all depots
    """
    total_weight = sum(c.weight for c in customers)
    total_capacity = sum(t.max_capacity for t in trucks)
    
    if total_weight > total_capacity:
        raise ValueError(
            f"Total customer weight ({total_weight}) exceeds total truck capacity ({total_capacity})"
        )
    
    depot_positions = np.array([d.location() for d in depots])
    customer_positions = np.array([c.location() for c in customers])
    
    min_distances_to_depot = []
    for cust_pos in customer_positions:
        min_dist = float('inf')
        for depot_pos in depot_positions:
            dist = np.sqrt((cust_pos[0] - depot_pos[0])**2 + (cust_pos[1] - depot_pos[1])**2)
            min_dist = min(min_dist, dist)
        min_distances_to_depot.append(min_dist)
    
    lower_bound = 2.0 * sum(min_distances_to_depot)
    
    upper_bound = lower_bound * 1.5
    
    return lower_bound, upper_bound
