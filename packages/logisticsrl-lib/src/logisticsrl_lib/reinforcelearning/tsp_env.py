import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces

from loader_lib.data_loader import FleetStatus, TruckState
from logisticsrl_lib.reinforcelearning.rewards import NormalizedRewards

class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(       
        self,
        cfg,
        fleetStatus: FleetStatus,
        normalized_rewards: NormalizedRewards
       
    ):
        super().__init__()
        self.fleetStatus = fleetStatus
        self.normalized_rewards = normalized_rewards
        self.cfg = cfg
        self.num_nodes = self.fleetStatus.num_nodes()

        self.source_mask = self.fleetStatus.source_mask
        self.target_mask = ~self.source_mask 
        self.num_trucks = self.fleetStatus.num_trucks()
        
        min_vals = self.fleetStatus.nodes.min(dim=0).values.numpy()
        max_vals = self.fleetStatus.nodes.max(dim=0).values.numpy()
        low = np.tile(min_vals, (self.num_nodes, 1))
        high = np.tile(max_vals, (self.num_nodes, 1))
        
        # ---------- Observation space ----------
        self.observation_space = spaces.Dict({
            # Node coordinates
            "nodes": spaces.Box(
                low=low, high=high, shape=(self.num_nodes, 2), dtype=np.float32
            ),
            # Which nodes are targets
            "is_target": spaces.MultiBinary(self.num_nodes
            ),
            # Visited targets 
            "visited_targets": spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8)
            ,
            # Current position of each truck
            "truck_positions": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks)
            ,
            # Mask for inactive trucks
            "inactive_trucks_mask": spaces.MultiBinary(self.num_trucks),
            # Distance matrix
            "time_matrix": spaces.Box(
                low=0, high=np.inf, shape=(self.num_nodes, self.num_nodes), dtype=np.float32
            ),
            # Add truck_starts to observation space
            "truck_starts": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks),
            # Add total_time of each truck to observation space
            "truck_times": spaces.Box(
                low=0, high=np.inf, shape=(self.num_trucks,), dtype=np.float32
            )
        })
        


        # ---------- Action space ----------
        # the total size is num_nodes x num_trucks .  +1 for NO-OP
        self.action_space = spaces.MultiDiscrete([self.num_trucks, self.num_nodes + 1])
        #self.action_space = spaces.Discrete(self.num_nodes+ 1)

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
        
        # Initialize inactive_trucks_mask as a tensor of all False values, meaning all trucks are initially active
        self.inactive_trucks_mask = torch.zeros(len(self.fleetStatus.trucklist), dtype=torch.bool).to(self.cfg.device)
 


        return self._get_obs(), {}

    def _get_obs(self):
        return  {
            "nodes": self.fleetStatus.nodes.numpy(),
            "is_target": self.target_mask.astype(np.int8),
            "visited_targets": self.visited_targets.astype(np.int8),
            "truck_positions": self.fleetStatus.truck_positions().copy(), #copy to avoid reference issues
            "inactive_trucks_mask": self.inactive_trucks_mask.cpu().numpy(),
            "time_matrix": self.fleetStatus.time_matrix.float(),
            "truck_starts": self.fleetStatus.truck_starts.copy(),
            "truck_times": self.fleetStatus.truck_times().copy(),
        }

    def step(self, action):
        
        truck_id, selected_node  = action
        
        self.num_steps += 1
        terminated = False        
        prev_node = self.fleetStatus.trucklist[truck_id].position
        reward = 0.0
        if selected_node==self.num_nodes: # NO-OP action
            
            reward += self.normalized_rewards.getRewardNonOP()
            self.inactive_trucks_mask[truck_id] = True  # Mark the selected truck as inactive
        else:
            reward += self.normalized_rewards.getRewardVisitBonus()
            
            self.fleetStatus.trucklist[truck_id].position = selected_node
            self.visited_targets[selected_node] = True        
            self.fleetStatus.trucklist[truck_id].tour.append(selected_node)
            
            dist = self.fleetStatus.time_matrix[prev_node, selected_node]
            reward += self.normalized_rewards.getRewardDistance(prev_node, selected_node)
            self.fleetStatus.trucklist[truck_id].total_time += dist
                   
        done = self.visited_targets[self.target_mask].all()

        if self.num_steps >= self.num_nodes + self.cfg.max_extra_steps:
            terminated = True
                            
        if done or terminated:
            reward += self.normalized_rewards.getRewardTotalFleetTime(self.fleetStatus.total_fleet_time())
            
        return self._get_obs(), reward, done, terminated, {}


