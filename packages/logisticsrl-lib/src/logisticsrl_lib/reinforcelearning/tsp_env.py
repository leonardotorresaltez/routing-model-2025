import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from loader_lib.data_loader import FleetStatus, TruckState


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
        self.target_indices = np.where(self.target_mask)[0]
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
            "current_trucks": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks),            
            "action_mask": spaces.Box(
                low=0, high=1, 
                shape=(self.num_nodes + 1,),  # +1 for NO-OP
                dtype=np.uint8
            )
        })

        # ---------- Action space ----------
        # the total size is num_nodes x num_trucks .  +1 for NO-OP
        # self.action_space = spaces.Discrete(self.num_nodes+ 1)
        
        # Action space: choose (truck, customer) pair
        self.action_space = spaces.Discrete(self.num_trucks * self.num_nodes)

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

    def _get_obs(self) -> dict:
        # If episode is already terminated/truncated, return partial observation
        if self.fleetStatus.active_truck == -1:
            return {
                "nodes": self.fleetStatus.nodes.numpy(),
                "is_target": self.target_mask.astype(np.int8),
                "visited_targets": self.visited_targets.astype(np.int8),
                "current_trucks": self.fleetStatus.truck_positions().copy(),
                "action_mask": self._compute_action_mask()  # All valid (won't be used)
            }
        
        return {
            "nodes": self.fleetStatus.nodes.numpy(),
            "is_target": self.target_mask.astype(np.int8),
            "visited_targets": self.visited_targets.astype(np.int8),
            "current_trucks": self.fleetStatus.truck_positions().copy(),
            "action_mask": self._compute_action_mask()  # Compute only if active truck exists
        }
        
    def _compute_action_mask(self) -> np.ndarray:
        """
        Mask where 1 = INVALID, 0 = VALID
        """
        mask = np.zeros(self.num_trucks * self.num_nodes, dtype=np.uint8)
        
        # FIRST: Mask all depot/source nodes for all trucks
        for truck_id in range(self.num_trucks):
            for source_idx in np.where(self.source_mask)[0]:
                action_idx = truck_id * self.num_nodes + source_idx
                mask[action_idx] = 1  # Depots always masked
        
        # THEN: Mask customer nodes based on conditions
        for truck_id in range(self.num_trucks):
            truck = self.fleetStatus.trucklist[truck_id]
            depot_idx = self.fleetStatus.truck_starts[truck_id]
            max_time = self.cfg.max_daily_delivery_time_each_truck
            remaining_time = max_time - truck.total_time
            current_pos = truck.position
            
            for customer_idx in self.target_indices:
                action_idx = truck_id * self.num_nodes + customer_idx
                
                if self.visited_targets[customer_idx] == 1:
                    mask[action_idx] = 1
                    continue
                
                dist_to_customer = self.fleetStatus.time_matrix[current_pos, customer_idx]
                dist_customer_to_depot = self.fleetStatus.time_matrix[customer_idx, depot_idx]
                time_needed = dist_to_customer + dist_customer_to_depot
                
                if time_needed > remaining_time:
                    mask[action_idx] = 1
        
        return mask



    
    
    def _compute_valid_actions(self) -> np.ndarray:
        """
        Returns array of shape (num_nodes + 1) where:
        - 1 = action is valid
        - 0 = action is invalid
        """
        truck_id = self.fleetStatus.active_truck
        
        # Start with already-visited nodes (always invalid)
        mask = (~self.visited_targets).astype(np.uint8)
        
        # Also mask nodes that violate time constraints
        for node_idx in range(self.num_nodes):
            if mask[node_idx] == 1:  # Only check if not already masked
                dist = self.fleetStatus.time_matrix[
                    self.fleetStatus.trucklist[truck_id].position,
                    node_idx
                ]
                new_time = self.fleetStatus.trucklist[truck_id].total_time + dist
                
                # Mask if exceeds time limit
                if new_time > self.cfg.max_daily_delivery_time_each_truck:
                    mask[node_idx] = 0
        
        # NO-OP action (num_nodes index) always valid
        mask = np.append(mask, 1)
        
        return mask

    def step(self, action):
        """
        Action: (truck_id, customer_idx) pair
        Masking guarantees: truck can reach customer AND return to depot in time.
        """
        self.num_steps += 1
        truck_id = action // self.num_nodes     
        customer_idx = action % self.num_nodes 
        
        # Validation: ensure action is valid
        assert truck_id < self.num_trucks, f"truck_id {truck_id} >= {self.num_trucks}"
        assert customer_idx < self.num_nodes, f"customer_idx {customer_idx} >= {self.num_nodes}"
        assert self.visited_targets[customer_idx] == 0, f"Customer {customer_idx} already visited!"
    
        truncated = False        
        prev_node = self.fleetStatus.trucklist[truck_id].position
        reward = 0.0

        reward += 10.0  # Reward for visiting a new target
        self.fleetStatus.trucklist[truck_id].position = customer_idx
        self.visited_targets[customer_idx] = True
        self.fleetStatus.trucklist[truck_id].tour.append(customer_idx)
        
        dist = self.fleetStatus.time_matrix[prev_node, customer_idx]
        reward -= dist
        self.fleetStatus.trucklist[truck_id].total_time += dist
        
        done = self.visited_targets[self.target_mask].all()
        
        # Return to depot at end of episode
        
        
        # Check step limit (safety)
        truncated = self.num_steps >= self.num_nodes+500 # FIXME
        if truncated:
             print(f"DEBUG: Too many steps ({self.num_steps}). Terminating episode with penalty.")
             
        # Return all trucks to depot at end of episode (done OR truncated)
        if done or truncated:
            for t_id in range(self.num_trucks):
                current_pos = self.fleetStatus.trucklist[t_id].position
                depot_idx = self.fleetStatus.truck_starts[t_id]
                return_dist = self.fleetStatus.time_matrix[current_pos, depot_idx]
                self.fleetStatus.trucklist[t_id].total_time += return_dist
                # self.fleetStatus.trucklist[t_id].tour.append(depot_idx) 
        
        return self._get_obs(), reward, done, truncated, {}
        
        
            

        # # search for next truck that can act, if all exceed 24h, terminate episode
        # self.fleetStatus.active_truck, truncated = self._get_next_truck_id()  
        # if self.cfg.debug: 
        #     print("DEBUG: Next active truck:", self.fleetStatus.active_truck, "Terminated:", truncated)
        #     if (truncated): #TODO this is not posible because agent avoid it
        #         print(f"No more feasible actions. Terminating episode.")

        #unvisited_count = (self.visited_targets == False).sum().item()
        #reward -= (unvisited_count * 500.0) # Heavy penalty # Goal: maximize clients

       
        #avoid infinite loops: if too many steps, terminate episode with heavy penalty
        # if self.num_steps >= self.num_nodes+500:
        #     truncated = True
        #     reward -= 1000 # Heavy penalty for too many steps (to prevent infinite loops)
        #     if self.cfg.debug: print(f"DEBUG: Too many steps ({self.num_steps}). Terminating episode with penalty.")
        return self._get_obs(), reward, done, truncated, {}  # CORRECT GYMNASIUM ORDER
    
    
    def _get_next_truck_id(self):
        next_truck = (self.fleetStatus.active_truck + 1) % self.num_trucks
        for _ in range(self.num_trucks):
            current_time = self.fleetStatus.trucklist[next_truck].total_time
            times_to_other_nodes = self.fleetStatus.time_matrix[self.fleetStatus.trucklist[next_truck].position]  # Time to all nodes from current position
            coming_back_times = self.fleetStatus.time_matrix[:, self.fleetStatus.trucklist[next_truck].tour[0]]  # Time to return to depot from all nodes
            potential_times = current_time + times_to_other_nodes + coming_back_times
            visited_mask = torch.tensor(self.visited_targets).bool()
            min_potential_time = potential_times.masked_fill(visited_mask, float('inf')).min()
            if min_potential_time <= self.cfg.max_daily_delivery_time_each_truck:   #TODO this is not posible because agent avoid it, but we need to check it anyway
                if self.cfg.debug: print(f"DEBUG: Truck {next_truck} can act (current_time={current_time:.2f}h, min_potential_time={min_potential_time:.2f}h).")
                return next_truck, False
            next_truck = (next_truck + 1) % self.num_trucks
        # If all trucks exceed 24h, return -1 and terminated
        return -1, True
