"""
Simplified Fleet Routing Environment using Gymnasium.

This environment simulates a fleet routing problem where:
- Multiple trucks deliver goods to customers from their home depots
- Objective: minimize distance + maximize truck utilization
- No time windows, no packing complexity
- Weight capacity is the only constraint

Environment:
- State: truck positions, loads, customer locations, feasibility mask
- Action: assign one customer (or depot return) to each truck per step
- Reward: distance cost + utilization bonus
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.types_simple import Customer, Depot, Truck, TruckState
from env.utils_simple import (build_distance_matrix, can_truck_serve_customer,
                              compute_utilization_reward, euclidean_distance)


class SimpleFleetRoutingEnv(gym.Env):
    """
    Gymnasium environment for fleet routing optimization.
    
    What it does:
    Simulates the world (trucks delivering customers)
    Provides observations (truck positions, customer locations)
    Computes rewards (distance cost, utilization bonus)
    Enforces constraints (feasibility masks)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        customers: List[Customer],
        trucks: List[Truck],
        depots: List[Depot],
        max_steps: int = 1000
    ):
        """
        Initialize the routing environment.
        
        Args:
            customers: list of Customer objects (delivery locations)
            trucks: list of Truck objects
            depots: list of Depot objects (home bases)
            max_steps: maximum steps per episode
        """
        super().__init__()
        
        self.customers = customers
        self.trucks = trucks
        self.depots = depots
        self.max_steps = max_steps
        
        self.num_customers = len(customers)
        self.num_trucks = len(trucks)
        self.num_depots = len(depots)
        
        # Precompute distance matrix for efficiency
        self.distance_matrix = build_distance_matrix(customers, depots, trucks)
        
        # Episode state
        self.truck_states: List[TruckState] = []
        self.current_step = 0
        self.episode_distance = 0.0
        self.episode_rewards = []
        
        # Gymnasium spaces
        # Action: MultiDiscrete, one action per truck [0 to N] where N=depot action
        self.action_space = spaces.MultiDiscrete([self.num_customers + 1] * self.num_trucks)
        
        # Observation: dictionary with various state components
        self.observation_space = spaces.Dict({
            "truck_positions": spaces.Box(low=-1000, high=1000, shape=(self.num_trucks, 2), dtype=np.float32),
            "truck_loads": spaces.Box(low=0, high=10000, shape=(self.num_trucks,), dtype=np.float32),
            "truck_capacities": spaces.Box(low=0, high=10000, shape=(self.num_trucks,), dtype=np.float32),
            "customer_locations": spaces.Box(low=-1000, high=1000, shape=(self.num_customers, 2), dtype=np.float32),
            "customer_weights": spaces.Box(low=0, high=1000, shape=(self.num_customers,), dtype=np.float32),
            "unvisited_mask": spaces.Box(low=0, high=1, shape=(self.num_customers,), dtype=np.int8),
            "feasibility_mask": spaces.Box(low=0, high=1, shape=(self.num_trucks, self.num_customers + 1), dtype=np.int8),
        })
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            obs: initial observation
            info: metadata dictionary
        """
        super().reset(seed=seed)
        
        # Reset episode counters
        self.current_step = 0
        self.episode_distance = 0.0
        self.episode_rewards = []
        
        # Initialize truck states
        self.truck_states = []
        for truck in self.trucks:
            home_depot = self.depots[truck.depot_id]
            state = TruckState(
                truck=truck,
                current_location=home_depot.location(),
                current_load=0.0,
                visited_customers=[]
            )
            self.truck_states.append(state)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: array of length num_trucks, each element in [0, num_customers]
                   where num_customers means "return to home depot"
        
        Returns:
            obs: observation after step
            reward: total reward (distance penalty + utilization bonus)
            terminated: True if all customers delivered and trucks home
            truncated: True if max_steps exceeded
            info: metadata
        """
        self.current_step += 1
        
        # Convert action to ensure valid types
        action = np.asarray(action, dtype=int)
        
        step_distance_cost = 0.0
        step_utilization_reward = 0.0
        
        # Process each truck's action
        for truck_idx, action_val in enumerate(action):
            truck_state = self.truck_states[truck_idx]
            
            if action_val == self.num_customers:
                # Action: return to home depot
                home_depot = self.depots[truck_state.truck.depot_id]
                distance = euclidean_distance(truck_state.current_location, home_depot.location())
                
                step_distance_cost += distance
                self.episode_distance += distance
                
                # Reset truck at depot
                truck_state.current_location = home_depot.location()
                truck_state.current_load = 0.0
                truck_state.visited_customers = []
            
            else:
                # Action: visit customer
                customer_idx = int(action_val)
                
                if customer_idx < self.num_customers and customer_idx not in truck_state.visited_customers:
                    customer = self.customers[customer_idx]
                    
                    # Check capacity
                    if can_truck_serve_customer(truck_state, customer):
                        # Calculate distance and update state
                        distance = euclidean_distance(truck_state.current_location, customer.location())
                        
                        step_distance_cost += distance
                        self.episode_distance += distance
                        
                        # Update truck state
                        truck_state.current_location = customer.location()
                        truck_state.current_load += customer.weight
                        truck_state.visited_customers.append(customer_idx)
        
        # Calculate rewards
        # r_routing: negative distance cost (minimize distance)
        r_routing = -step_distance_cost
        
        # r_efficiency: utilization bonus
        r_efficiency = compute_utilization_reward(self.truck_states)
        
        # Total reward: 0.75 on routing, 0.25 on efficiency
        reward = 0.75 * r_routing + 0.25 * r_efficiency
        self.episode_rewards.append(reward)
        
        # Check termination: all customers visited
        all_customers_visited = all(
            len(ts.visited_customers) > 0 or ts.current_load == 0
            for ts in self.truck_states
        )
        
        # Better termination: all customers have been served by someone
        all_customers_served = set()
        for truck_state in self.truck_states:
            all_customers_served.update(truck_state.visited_customers)
        
        customers_delivered = len(all_customers_served)
        terminated = (customers_delivered == self.num_customers)
        
        # Check truncation: max steps exceeded
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Compute current observation from environment state.
        
        Returns:
            Dictionary with observation components
        """
        # Truck positions
        truck_positions = np.array([ts.current_location for ts in self.truck_states], dtype=np.float32)
        
        # Truck loads
        truck_loads = np.array([ts.current_load for ts in self.truck_states], dtype=np.float32)
        
        # Truck capacities
        truck_capacities = np.array([ts.truck.max_capacity for ts in self.truck_states], dtype=np.float32)
        
        # Customer locations
        customer_locations = np.array([c.location() for c in self.customers], dtype=np.float32)
        
        # Customer weights
        customer_weights = np.array([c.weight for c in self.customers], dtype=np.float32)
        
        # Unvisited mask
        all_visited = set()
        for ts in self.truck_states:
            all_visited.update(ts.visited_customers)
        
        unvisited_mask = np.array(
            [1 if i not in all_visited else 0 for i in range(self.num_customers)],
            dtype=np.int8
        )
        
        # Feasibility mask: (num_trucks, num_customers + 1)
        # Column i: customer i (if i < num_customers)
        # Column num_customers: depot return action
        feasibility_mask = np.zeros((self.num_trucks, self.num_customers + 1), dtype=np.int8)
        
        for truck_idx, truck_state in enumerate(self.truck_states):
            # Can always return to depot
            feasibility_mask[truck_idx, self.num_customers] = 1
            
            # Can visit customer if: unvisited AND has capacity AND not already visited
            for customer_idx in range(self.num_customers):
                if customer_idx in all_visited:
                    # Already visited by some truck
                    feasibility_mask[truck_idx, customer_idx] = 0
                elif customer_idx in truck_state.visited_customers:
                    # Already visited by THIS truck (shouldn't happen in our logic)
                    feasibility_mask[truck_idx, customer_idx] = 0
                elif can_truck_serve_customer(truck_state, self.customers[customer_idx]):
                    # Has capacity
                    feasibility_mask[truck_idx, customer_idx] = 1
                else:
                    # No capacity
                    feasibility_mask[truck_idx, customer_idx] = 0
        
        return {
            "truck_positions": truck_positions,
            "truck_loads": truck_loads,
            "truck_capacities": truck_capacities,
            "customer_locations": customer_locations,
            "customer_weights": customer_weights,
            "unvisited_mask": unvisited_mask,
            "feasibility_mask": feasibility_mask,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dictionary with episode metrics.
        
        Returns:
            Dictionary with various metrics
        """
        all_visited = set()
        for ts in self.truck_states:
            all_visited.update(ts.visited_customers)
        
        avg_utilization = np.mean([ts.utilization() for ts in self.truck_states])
        
        return {
            "step": self.current_step,
            "total_distance": self.episode_distance,
            "customers_delivered": len(all_visited),
            "avg_utilization": avg_utilization,
            "total_reward": sum(self.episode_rewards),
        }
    
    def render(self):
        """Render not implemented for this simplified version."""
        pass
    
    def close(self):
        """Close environment."""
        pass
