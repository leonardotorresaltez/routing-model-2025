"""
Baseline policies for fleet routing.

Implements simple heuristics (greedy, nearest-neighbor) for benchmarking
against learned RL policies.
"""

from typing import Any, Dict, List

import numpy as np

from env.routing_env_simple import SimpleFleetRoutingEnv
from env.utils_simple import euclidean_distance


def greedy_nearest_customer_policy(env: SimpleFleetRoutingEnv) -> Dict[str, Any]:
    """
    Greedy policy: each truck picks nearest unvisited customer it can serve.
    
    If a truck cannot serve any customer (capacity constraint), it returns
    to depot. At each step:
    1. For each truck, find all feasible customers (capacity + unvisited)
    2. Pick the nearest feasible customer
    3. If no feasible customer, return to depot
    
    Args:
        env: SimpleFleetRoutingEnv instance
    
    Returns:
        Dictionary with episode metrics
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        action = np.zeros(env.num_trucks, dtype=int)
        
        for truck_idx in range(env.num_trucks):
            truck_state = env.truck_states[truck_idx]
            truck_pos = truck_state.current_location
            
            best_customer = None
            best_distance = float('inf')
            
            all_visited = set()
            for ts in env.truck_states:
                all_visited.update(ts.visited_customers)
            
            for customer_idx in range(env.num_customers):
                if customer_idx in all_visited:
                    continue
                if customer_idx in truck_state.visited_customers:
                    continue
                
                customer = env.customers[customer_idx]
                if truck_state.remaining_capacity() >= customer.volume:
                    dist = euclidean_distance(truck_pos, customer.location())
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer_idx
            
            if best_customer is not None:
                action[truck_idx] = best_customer
            else:
                action[truck_idx] = env.num_customers
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    return {
        "total_reward": total_reward,
        "episode_length": step_count,
        "total_distance": info["total_distance"],
        "customers_delivered": info["customers_delivered"],
        "avg_utilization": info["avg_utilization"],
    }


def nearest_depot_first_policy(env: SimpleFleetRoutingEnv) -> Dict[str, Any]:
    """
    Conservative policy: truck returns to depot after each few deliveries.
    
    Trucks try to fill up to a threshold utilization (e.g., 70%) before
    returning to depot. If cannot serve any customer, return immediately.
    
    Args:
        env: SimpleFleetRoutingEnv instance
    
    Returns:
        Dictionary with episode metrics
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    utilization_threshold = 0.7
    
    while not done:
        action = np.zeros(env.num_trucks, dtype=int)
        
        for truck_idx in range(env.num_trucks):
            truck_state = env.truck_states[truck_idx]
            truck_pos = truck_state.current_location
            
            utilization = truck_state.utilization()
            
            if utilization >= utilization_threshold:
                action[truck_idx] = env.num_customers
                continue
            
            best_customer = None
            best_distance = float('inf')
            
            all_visited = set()
            for ts in env.truck_states:
                all_visited.update(ts.visited_customers)
            
            for customer_idx in range(env.num_customers):
                if customer_idx in all_visited:
                    continue
                if customer_idx in truck_state.visited_customers:
                    continue
                
                customer = env.customers[customer_idx]
                if truck_state.remaining_capacity() >= customer.volume:
                    dist = euclidean_distance(truck_pos, customer.location())
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer_idx
            
            if best_customer is not None:
                action[truck_idx] = best_customer
            else:
                action[truck_idx] = env.num_customers
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    return {
        "total_reward": total_reward,
        "episode_length": step_count,
        "total_distance": info["total_distance"],
        "customers_delivered": info["customers_delivered"],
        "avg_utilization": info["avg_utilization"],
    }


def furthest_customer_policy(env: SimpleFleetRoutingEnv) -> Dict[str, Any]:
    """
    Exploratory policy: each truck picks furthest unvisited customer.
    
    This encourages spreading out and covering the entire area,
    potentially useful for initial exploration.
    
    Args:
        env: SimpleFleetRoutingEnv instance
    
    Returns:
        Dictionary with episode metrics
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        action = np.zeros(env.num_trucks, dtype=int)
        
        for truck_idx in range(env.num_trucks):
            truck_state = env.truck_states[truck_idx]
            truck_pos = truck_state.current_location
            
            best_customer = None
            best_distance = -1.0
            
            all_visited = set()
            for ts in env.truck_states:
                all_visited.update(ts.visited_customers)
            
            for customer_idx in range(env.num_customers):
                if customer_idx in all_visited:
                    continue
                if customer_idx in truck_state.visited_customers:
                    continue
                
                customer = env.customers[customer_idx]
                if truck_state.remaining_capacity() >= customer.volume:
                    dist = euclidean_distance(truck_pos, customer.location())
                    if dist > best_distance:
                        best_distance = dist
                        best_customer = customer_idx
            
            if best_customer is not None:
                action[truck_idx] = best_customer
            else:
                action[truck_idx] = env.num_customers
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    return {
        "total_reward": total_reward,
        "episode_length": step_count,
        "total_distance": info["total_distance"],
        "customers_delivered": info["customers_delivered"],
        "avg_utilization": info["avg_utilization"],
    }
