import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces

from core.utils.data_loader import FleetStatus, TruckState


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(       
        self,
        cfg,
        fleetStatus: FleetStatus,
       
    ):
        super().__init__()
        self.fleetStatus = fleetStatus
        self.cfg = cfg
        self.num_nodes = self.fleetStatus.num_nodes()

        self.source_mask = self.fleetStatus.source_mask
        self.target_mask = ~self.source_mask 
        self.num_trucks = self.fleetStatus.num_trucks()
        
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
            "visited_targets": spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8)
            ,
            # Current position of each truck
            "current_trucks": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks)
        })

        # ---------- Action space ----------
        # the total size is num_nodes x num_trucks .  +1 for NO-OP
        self.action_space = spaces.Discrete(self.num_nodes+ 1)

        self.reset()        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        self.fleetStatus.trucklist = {
            i: TruckState(total_time=0.0, tour=[self.fleetStatus.truck_starts[i]], position=self.fleetStatus.truck_starts[i])
            for i in range(len(self.fleetStatus.truck_starts))
        }       
        self.num_steps = 0        
        self.visited_targets = np.zeros(self.num_nodes, dtype=np.int8)  # 0 = not visited, 1 = visited    
        self.visited_targets[self.source_mask] = True      
        
        # truck to act
        self.fleetStatus.active_truck = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return  {
            "nodes": self.fleetStatus.nodes.numpy(),
            "is_target": self.target_mask.astype(np.int8),
            "visited_targets": self.visited_targets.astype(np.int8),
            "current_trucks": self.fleetStatus.truck_positions().copy(), #copy to avoid reference issues
        }

    def step(self, action):
        self.num_steps += 1
        truck_id = self.fleetStatus.active_truck       
        terminated = False        
        prev_node = self.fleetStatus.trucklist[truck_id].position
        reward = 0.0
        if action==self.num_nodes: # NO-OP action
            reward -=  100.0  # Heavy penalty for NO-OP to encourage visiting customers
        else:
            reward += 10.0  # Reward for visiting a new target
            self.fleetStatus.trucklist[truck_id].position = action
            self.visited_targets[action] = True        
            self.fleetStatus.trucklist[truck_id].tour.append(action)
            
            dist = self.fleetStatus.time_matrix[prev_node, action]
            reward -= dist
            self.fleetStatus.trucklist[truck_id].total_time += dist
        
            
        done = self.visited_targets[self.target_mask].all()

        # search for next truck that can act, if all exceed 24h, terminate episode
        self.fleetStatus.active_truck, terminated = self._get_next_truck_id()  
        if (terminated): #TODO this is not posible because agent avoid id
            print(f"All trucks exceeded 24h xxxx. Terminating episode.")

        unvisited_count = (self.visited_targets == False).sum().item()
        reward -= (unvisited_count * 500.0) # Heavy penalty # Goal: maximize clients

       
        #avoid infinite loops: if too many steps, terminate episode with heavy penalty
        if self.num_steps >= self.num_nodes+500:
            terminated = True
            reward -= 1000 # Heavy penalty for too many steps (to prevent infinite loops)
        return self._get_obs(), reward, done, terminated, {}
    
    
    def _get_next_truck_id(self):
        next_truck = (self.fleetStatus.active_truck + 1) % self.num_trucks
        for _ in range(self.num_trucks):
            if self.fleetStatus.trucklist[next_truck].total_time <= 24:
                return next_truck, False
            next_truck = (next_truck + 1) % self.num_trucks
        # If all trucks exceed 24h, return -1 and terminated
        return -1, True
