import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces

class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.num_nodes = cfg.num_nodes
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(0.0, 1.0, (self.num_nodes, 2), dtype=np.float32),
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