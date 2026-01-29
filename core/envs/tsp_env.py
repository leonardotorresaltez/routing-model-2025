import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, nodes, current):
        super().__init__()
#        self.num_nodes = cfg.num_nodes
#        self.action_space = spaces.Discrete(self.num_nodes)
#        self.observation_space = spaces.Dict({
#            "nodes": spaces.Box(0.0, 1.0, (self.num_nodes, 2), dtype=np.float32), # TODO: Fixed nodes from a csv
#            # TODO: Take adjacency distance matrix from a csv
#            "current": spaces.Discrete(self.num_nodes),
#            "visited": spaces.MultiBinary(self.num_nodes)
#        })
        self.num_sources = cfg.num_sources
        self.num_targets = cfg.num_targets
        self.num_trucks = cfg.num_trucks
        self.num_nodes = cfg.num_sources + cfg.num_targets
        self.nodes = nodes
        self.current = current  # list of current positions for each truck
        
        # ---------- Observation space ----------
        self.observation_space = spaces.Dict({
            # Node coordinates
            "nodes": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_nodes, 2), dtype=np.float32
            ),
            # Which nodes are targets
            "is_target": spaces.Box(
                low=0, high=1, shape=(self.num_nodes,), dtype=np.int8
            ),
            # Visited targets (GLOBAL)
            "visited_targets": spaces.Box(
                low=0, high=1, shape=(self.num_nodes,), dtype=np.int8
            ),
            # Current position of each truck
            "current": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks)
        })

        # ---------- Action space ----------
        # One action per truck
        self.action_space = spaces.MultiDiscrete(
            [self.num_nodes] * self.num_trucks
        )

        self.reset()        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        
        # Labels
        #example tensor([False, False, False, False, False])
        self.is_source = torch.zeros(self.num_nodes, dtype=torch.bool)
        #example if num_sources=2, tensor([True, True, False, False, False])
        self.is_source[:self.num_sources] = True

        #example tensor([False, False, True, True, True])
        self.is_target = ~self.is_source

        # Initial truck positions (sources) , torch.arange generate unidimensional tensor with consecutive integers
        #example  tensor([0, 1])
        #source_indices = torch.arange(self.num_sources)
        


        # Visited targets (GLOBAL)
        #example tensor([False, False, False, False, False]), self.nodes = 5
        self.visited_targets = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        #example tensor([True, True, False, False, False])
        #sources are marked as visited from the start 
        #is_source works as a mask here, in each index where is_source is True, visited_targets is set to True
        self.visited_targets[self.is_source] = True  # sources are irrelevant
        
        
        return self._get_obs(), {}

    def _get_obs(self):
        return  {
            "nodes": self.nodes.clone(),
            "is_target": self.is_target.clone().int(),
            "visited_targets": self.visited_targets.clone().int(),
            "current": np.array(self.current, dtype=np.int64)
        }

    def step(self, actions):
        reward = 0.0

        for k, action in enumerate(actions):
            prev = self.current[k]
            self.current[k] = action

            # Mark target as visited
            self.visited_targets[action] = True

            dist = torch.norm(self.nodes[prev] - self.nodes[action])
            reward -=  10.0 * dist  # Negative reward = distance cost

        done = self.visited_targets[self.is_target].all()
        return self._get_obs(), reward, done, False, {}
    
    
    

#source_indices = torch.arange(5)   
#print(source_indices)
##tensor([0, 1])
#aux = torch.randperm(5)[:3]
#print(aux)
#source_indices = source_indices[aux].tolist()
#print(source_indices)