import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(       
        self,
        cfg,
        nodes,                    # Tensor [N, 2]
        source_mask,              # np.array [N] bool
        truck_starts,  # list[int]
        time_matrix,         
    ):
        super().__init__()
        self.cfg = cfg
        self.time_matrix = time_matrix # Store the actual travel times
        self.nodes = nodes.clone() #then enviroment has ther own copy
        self.num_nodes = nodes.shape[0]

        self.source_mask = source_mask
        self.target_mask = ~source_mask

        self.truck_starts = list(truck_starts) #safe cast for list
        self.num_trucks = len(truck_starts)
        
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
        # the total size is num_nodes x num_trucks
        self.action_space = spaces.Discrete(self.num_nodes)


        self.reset()        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        # reset total_time_bytruck
        self.total_time_bytruck = [0.0 for _ in range(self.num_trucks)]        
        
        # reset truck positions (fixed)
        self.truck_positions = np.array(
            self.truck_starts, dtype=np.int64
        )   
        
        # visited mask
        self.visited_targets = np.zeros(self.num_nodes, dtype=bool)             
      
        # sources are considered visited
        self.visited_targets[self.source_mask] = True      
        
        # truck to act
        self.active_truck = 0
        
        # tours start with initial positions
        self.tours = [[pos] for pos in self.truck_starts]

        # Reset total_time_bytruck
        self.total_time_bytruck = [0.0 for _ in range(self.num_trucks)]
        
        return self._get_obs(), {}

    def _get_obs(self):
        return  {
            "nodes": self.nodes.numpy(),
            "is_target": self.target_mask.astype(np.int8),
            "visited_targets": self.visited_targets.astype(np.int8),
            "current_trucks": self.truck_positions.copy(), #copy to avoid reference issues
        }

    def step(self, action):
        truck_id = self.active_truck
        prev_node = self.truck_positions[truck_id]

        self.truck_positions[truck_id] = action
        self.visited_targets[action] = True
        
        self.tours[truck_id].append(action)

        dist = self.time_matrix[prev_node, action]
        reward = -dist
        
        # acumulate time for the truck
        self.total_time_bytruck[truck_id] += dist

        # Buscar el siguiente camión disponible (que no supere 24h)
        #TODO more options to a better moving between trucks .. choose next truck also use masks 
        next_truck = (self.active_truck + 1) % self.num_trucks
        for _ in range(self.num_trucks):
            if self.total_time_bytruck[next_truck] <= self.cfg.max_daily_delivery_time_each_truck:
                self.active_truck = next_truck
                break
            next_truck = (next_truck + 1) % self.num_trucks
        
        # Terminate solo si todos los camiones superaron el límite
        terminated = all(t > self.cfg.max_daily_delivery_time_each_truck for t in self.total_time_bytruck)        
        
        done = self.visited_targets[self.target_mask].all()



        return self._get_obs(), reward, done, terminated, {}
    
    
    
