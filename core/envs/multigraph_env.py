import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces

class MultiGraphEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_nodes = cfg.num_nodes
        self.num_trucks = cfg.num_trucks
        self.depot = cfg.depot
        self.max_steps = cfg.max_steps

        # Each truck chooses a node
        self.action_space = spaces.MultiDiscrete([self.num_nodes] * self.num_trucks)

        # Observation space
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(0.0, 1.0, (self.num_nodes, 3), dtype=np.float32),
            "edge_index": spaces.Box(0, self.num_nodes - 1,
                                     (2, self.num_nodes * (self.num_nodes - 1)),
                                     dtype=np.int64),
            "edge_attr": spaces.Box(0.0, 1.5,
                                    (self.num_nodes * (self.num_nodes - 1), 1),
                                    dtype=np.float32),
            "truck_pos": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks),
            "visited": spaces.MultiBinary(self.num_nodes)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Node coordinates
        self.nodes = torch.rand(self.num_nodes, 2)

        # Trucks start at depot
        self.truck_pos = torch.zeros(self.num_trucks, dtype=torch.long)

        # Visited mask
        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited[self.depot] = True

        self.steps = 0
        self._build_graph()

        return self._get_state(), {}

    def _build_graph(self):
        visited_float = self.visited.float().unsqueeze(1)
        self.node_features = torch.cat([self.nodes, visited_float], dim=1)

        senders, receivers, edge_attr = [], [], []

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                senders.append(i)
                receivers.append(j)
                dist = torch.norm(self.nodes[i] - self.nodes[j])
                edge_attr.append(dist.item())

        self.edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(1)

    def _update_node_features(self):
        visited_float = self.visited.float().unsqueeze(1)
        self.node_features = torch.cat([self.nodes, visited_float], dim=1)

    def _get_state(self):
        self._update_node_features()
        return {
            "node_features": self.node_features.clone(),
            "edge_index": self.edge_index.clone(),
            "edge_attr": self.edge_attr.clone(),
            "truck_pos": self.truck_pos.clone(),
            "visited": self.visited.clone()
        }

    def step(self, action):
        action = torch.tensor(action, dtype=torch.long)
        total_reward = 0.0
        self.steps += 1

        for t in range(self.num_trucks):
            prev = self.truck_pos[t]
            nxt = action[t]
            self.truck_pos[t] = nxt

            dist = torch.norm(self.nodes[prev] - self.nodes[nxt])
            total_reward -= dist.item()

            # Reward for visiting new vendor
            if not self.visited[nxt]:
                total_reward += 10.0

            # Reward for returning to depot
            if nxt == self.depot:
                total_reward += 5.0

            self.visited[nxt] = True

        all_vendors_visited = self.visited[1:].all().item()
        all_trucks_at_depot = (self.truck_pos == self.depot).all().item()

        terminated = bool(all_vendors_visited and all_trucks_at_depot)
        truncated = False

        if self.steps >= self.max_steps:
            truncated = True
            total_reward -= 50.0

        return self._get_state(), total_reward, terminated, truncated, {}