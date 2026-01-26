import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.num_nodes = cfg.num_nodes
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(0.0, 1.0, (self.num_nodes, 2), dtype=np.float32), # TODO: Fixed nodes from a csv
            # TODO: Take adjacency distance matrix from a csv
            "current": spaces.Discrete(self.num_nodes),
            "visited": spaces.MultiBinary(self.num_nodes)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.nodes = torch.rand(self.num_nodes, 2)
        self.current = random.randrange(self.num_nodes)
        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited[self.current] = True
        self.tour = [self.current]
        return self._get_state(), {}

    def _get_state(self):
        return {
            "nodes": self.nodes.clone(),
            "current": self.current,
            "visited": self.visited.clone()
        }

    def step(self, action):
        prev = self.current
        self.current = action
        self.visited[action] = True
        self.tour.append(action)

        dist = torch.norm(self.nodes[prev] - self.nodes[action])
        reward = -dist # Minimize distance = Maximize negative distance

        terminated = self.visited.all()
        return self._get_state(), reward, terminated, False, {}
    
    
    




class MDVRPEnv(gym.Env):
    """ 
    single-step reward calculation based on the total travel time of all 50 trucks.
    
    Single-Step Journey: It calculates all 50 routes at once and returns terminated=True.
    No Euclidean Norm: It uses the actual time_matrix values from your files.
    Multi-Depot Return: Each truck returns to its specific depot_idx.
    """
    def __init__(self, cfg, data):
        super().__init__()
        self.cfg = cfg
        # We use the time-proximity profiles as observations
        self.node_features = data["node_features"]
        self.time_matrix = data["time_matrix"]
        self.trucks = data["trucks"]
        self.depots = data["depots"]
        self.customers = data["customers"]
        self.num_nodes = data["num_nodes"]

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(0.0, 1.0, (self.num_nodes, self.num_nodes), dtype=np.float32)
        })
        
        # In a one-shot environment, the action space is handled logically by the agent
        self.action_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        """
        action: Dict[truck_id, List[customer_idx]]
        """
        total_time = 0.0
        visited_customers = set()
        
        for truck in self.trucks:
            route = action.get(truck.id, [])
            if not route:
                continue
            
            prev_idx = truck.depot_idx
            truck_time = 0.0
            
            for cust_idx in route:
                # Add time from the CSV-based time_matrix
                truck_time += self.time_matrix[prev_idx, cust_idx].item()
                visited_customers.add(cust_idx)
                prev_idx = cust_idx
            
            # Add time to return to the depot
            truck_time += self.time_matrix[prev_idx, truck.depot_idx].item()
            total_time += truck_time

        # Negative reward to minimize total time
        reward = -total_time
        
        # Penalty for skipping customers
        missing = len(self.customers) - len(visited_customers)
        reward -= missing * 50.0 # High penalty to ensure coverage
        
        # One-shot: always returns terminated=True
        return self._get_obs(), reward, True, False, {"total_time": total_time}

    def _get_obs(self):
        return {"node_features": self.node_features.clone()}
