import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg,nodes, truck_starts, time_matrix):
        super().__init__()
        self.num_nodes = cfg.num_nodes
        self.nodes = nodes
        self.truck_starts = truck_starts
        self.current_positions = truck_starts.copy()
        self.time_matrix = time_matrix # Store the actual travel times
        
        # ---- Observation space ----
        self.observation_space = spaces.Dict({
        "nodes": spaces.Box(low=0.0, high=1.0, shape=(self.num_nodes, 2), dtype=np.float32),
        "visited": spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8),
        "current": spaces.MultiDiscrete([self.num_nodes] * len(truck_starts))
        })        
        # ---- Action space ----
        self.action_space = spaces.Discrete(self.num_nodes)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.current_positions = self.truck_starts.copy()
        
        # Mark all depots as visited for all trucks
        for pos in self.truck_starts:
            self.visited[pos] = True
        
        self.tours = [[pos] for pos in self.truck_starts]
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "nodes": self.nodes.clone(),
            "visited": self.visited.clone().int(),
            "current": torch.tensor(self.current_positions, dtype=torch.long)
        }

   
    
    def step(self, action, truck_id):
        prev = self.current_positions[truck_id]
        self.current_positions[truck_id] = action
        
        self.visited[action] = True
        self.tours[truck_id].append(action)
        
        travel_time = self.time_matrix[prev, action]
        reward = -travel_time 

        terminated = self.visited.all()
        return self._get_obs(), reward, terminated, False, {}