import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        nodes,                    # Tensor [N, 2]
        source_mask,              # np.array [N] bool
        initial_truck_positions,  # list[int]
    ):
        super().__init__()


        self.nodes = nodes.clone()
        self.num_nodes = nodes.shape[0]

        self.source_mask = source_mask
        self.target_mask = ~source_mask

        self.initial_truck_positions = list(initial_truck_positions)
        self.num_trucks = len(initial_truck_positions)
        
        # ---------- Observation space ----------
        self.observation_space = spaces.Dict({
            # Node coordinates
            "nodes": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_nodes, 2), dtype=np.float32
            ),
            # Which nodes are targets
            "is_target": spaces.MultiBinary(self.num_nodes
            ),
            # Visited targets 
            "visited_targets": spaces.MultiBinary(self.num_nodes
            ),
            # Current position of each truck
            "current_trucks": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks)
        })

        # ---------- Action space ----------
        # 
        self.action_space = spaces.Discrete(self.num_nodes)

        self.reset()        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        # reset truck positions (fixed)
        self.truck_positions = np.array(
            self.initial_truck_positions, dtype=np.int64
        )   
        
        # visited mask
        self.visited_targets = np.zeros(self.num_nodes, dtype=bool)             
      
        # sources are considered visited
        self.visited_targets[self.source_mask] = True      
        
        
        self.active_truck = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        return  {
            "nodes": self.nodes.numpy(),
            "is_target": self.target_mask.astype(np.int8),
            "visited_targets": self.visited_targets.astype(np.int8),
            "current_trucks": self.truck_positions.copy(),
        }

    def step(self, action):
        truck_id = self.active_truck
        prev_node = self.truck_positions[truck_id]

        self.truck_positions[truck_id] = action
        self.visited_targets[action] = True

        dist = torch.norm(
            self.nodes[prev_node] - self.nodes[action]
        ).item()
        reward = -dist

        self.active_truck = (self.active_truck + 1) % self.num_trucks

        done = self.visited_targets[self.target_mask].all()

        return self._get_obs(), reward, done, False, {}
    
    
    

#source_indices = torch.arange(5)   
#print(source_indices)
##tensor([0, 1])
#aux = torch.randperm(5)[:3]
#print(aux)
#source_indices = source_indices[aux].tolist()
#print(source_indices)