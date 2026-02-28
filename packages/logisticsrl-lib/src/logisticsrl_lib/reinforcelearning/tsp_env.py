import gymnasium as gym
import torch
import random
import numpy as np
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
        
        self.fleetStatus.active_truck = 0

        self.finished_trucks = set() # to keep track of trucks that have decided no-op or have no feasible actions left
        self.noop_count = 0

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
        reward = 0
        if action==self.num_nodes: # NO-OP action
            # reward -= 1
            # unvisited_count = (self.visited_targets==0).sum().item()
            # reward -= (unvisited_count * 1) # Heavy penalty # Goal: maximize clients

            self.noop_count += 1
            self.finished_trucks.add(truck_id) # Mark this truck as finished (no more actions)
            self.fleetStatus.trucklist[truck_id].noop = True  # Mark the truck's state as having taken a NO-OP
            if self.cfg.debug: print(f"DEBUG: Truck {truck_id} takes NO-OP. Marking as finished.")
        else:
            reward += 1  # Reward for visiting a new target
            self.fleetStatus.trucklist[truck_id].position = action
            self.visited_targets[action] = True        
            self.fleetStatus.trucklist[truck_id].tour.append(action)
            
            dist = self.fleetStatus.time_matrix[prev_node, action]
            reward -= dist * self.cfg.distance_penalty_scale
            self.fleetStatus.trucklist[truck_id].total_time += dist
        
            
        done = self.visited_targets[self.target_mask].all()

        # search for next truck that can act, if all exceed 24h, terminate episode
        self.fleetStatus.active_truck, terminated = self._get_next_truck_id()  
        if self.cfg.debug: 
            print("DEBUG: Next active truck:", self.fleetStatus.active_truck, "Terminated:", terminated)
            if (terminated): #TODO this is not posible because agent avoid it
                print(f"No more feasible actions. Terminating episode.")

        
       
        #avoid infinite loops: if too many steps, terminate episode with heavy penalty
        if self.num_steps >= self.num_nodes+500:
            terminated = True
            reward -= 1000 # Heavy penalty for too many steps (to prevent infinite loops)
            if self.cfg.debug: print(f"DEBUG: Too many steps ({self.num_steps}). Terminating episode with penalty.")
        return self._get_obs(), reward, done, terminated, {}
    
    
    # def _get_next_truck_id(self):
    #     next_truck = (self.fleetStatus.active_truck + 1) % self.num_trucks
    #     for _ in range(self.num_trucks):
    #         if next_truck in self.finished_trucks:
    #             next_truck = (next_truck + 1) % self.num_trucks
    #             continue
    #         current_time = self.fleetStatus.trucklist[next_truck].total_time
    #         times_to_other_nodes = self.fleetStatus.time_matrix[self.fleetStatus.trucklist[next_truck].position]  # Time to all nodes from current position
    #         coming_back_times = self.fleetStatus.time_matrix[:, self.fleetStatus.trucklist[next_truck].tour[0]]  # Time to return to depot from all nodes
    #         potential_times = current_time + times_to_other_nodes + coming_back_times
    #         visited_mask = torch.tensor(self.visited_targets).bool()
    #         min_potential_time = potential_times.masked_fill(visited_mask, float('inf')).min()
    #         if min_potential_time <= self.cfg.max_daily_delivery_time_each_truck:   #TODO this is not posible because agent avoid it, but we need to check it anyway
    #             if self.cfg.debug: print(f"DEBUG: Truck {next_truck} can act (current_time={current_time:.2f}h, min_potential_time={min_potential_time:.2f}h).")
    #             return next_truck, False
    #         self.finished_trucks.add(next_truck)
    #         next_truck = (next_truck + 1) % self.num_trucks
    #     # If all trucks exceed 24h, return -1 and terminated
    #     return -1, True
    
    def _min_potential_delivery_time(self, truck_id):
        truck = self.fleetStatus.trucklist[truck_id]
        times_to_nodes = self.fleetStatus.time_matrix[truck.position]
        times_back_to_depot = self.fleetStatus.time_matrix[:, truck.tour[0]]
        potential_times = truck.total_time + times_to_nodes + times_back_to_depot

        visited_mask = torch.tensor(self.visited_targets).bool()
        return potential_times.masked_fill(visited_mask, float('inf')).min()


    def _truck_can_still_deliver(self, truck_id):
        return self._min_potential_delivery_time(truck_id) <= self.cfg.max_daily_delivery_time_each_truck
    
    def _get_next_truck_id(self):
        candidate = (self.fleetStatus.active_truck + 1) % self.num_trucks

        for _ in range(self.num_trucks):
            if candidate not in self.finished_trucks:
                if self._truck_can_still_deliver(candidate):
                    if self.cfg.debug:
                        print(f"DEBUG: Truck {candidate} can act (current_time={self.fleetStatus.trucklist[candidate].total_time:.2f}h, min_potential_time={self._min_potential_delivery_time(candidate):.2f}h).")
                    return candidate, False
                else:
                    self.finished_trucks.add(candidate)

            candidate = (candidate + 1) % self.num_trucks

        # All trucks have exceeded the max daily delivery time
        return -1, True
