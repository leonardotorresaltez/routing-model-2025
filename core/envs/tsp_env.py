import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces

class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg,nodes, current):
        super().__init__()
        self.num_nodes = cfg.num_nodes
        self.nodes = nodes
        self.current = current
        
        # ---- Observation space ----
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_nodes, 2), dtype=np.float32
            ),
            "visited": spaces.Box(
                low=0, high=1, shape=(self.num_nodes,), dtype=np.int8
            ),
            "current": spaces.Discrete(self.num_nodes)
        })
        self.current = current
        
        # ---- Action space ----
        self.action_space = spaces.Discrete(self.num_nodes)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited[self.current] = True
        self.tour = [self.current]
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "nodes": self.nodes.clone(),
            "visited": self.visited.clone().int(),
            "current": int(self.current)
        }

    def step(self, action):
        prev = self.current
        self.current = action
        self.visited[action] = True
        self.tour.append(action)

        dist = torch.norm(self.nodes[prev] - self.nodes[action])
        reward = -dist # Minimize distance = Maximize negative distance

        terminated = self.visited.all()
        return self._get_obs(), reward, terminated, False, {}