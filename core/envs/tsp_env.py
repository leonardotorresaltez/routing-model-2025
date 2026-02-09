import gymnasium as gym
import torch
import random
import numpy as np
from gymnasium import spaces

from core.utils.data_loader import TruckState


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
            "visited_targets": spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8)
            ,
            # Current position of each truck
            "current_trucks": spaces.MultiDiscrete([self.num_nodes] * self.num_trucks)
        })

        # ---------- Action space ----------
        # the total size is num_nodes x num_trucks
        self.action_space = spaces.Discrete(self.num_nodes+ 1)
        #self.action_space = spaces.Discrete(self.num_nodes)

        self.reset()        

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        # reset total_time_bytruck
        #self.total_time_bytruck = [0.0 for _ in range(self.num_trucks)]        
        self.num_steps = 0
        
#       self.trucks_active = [True for _ in range(self.num_trucks)]
        
        # reset truck positions (fixed)
        self.truck_positions = np.array(
            self.truck_starts, dtype=np.int64
        )   
        
        # visited mask
        self.visited_targets = np.zeros(self.num_nodes, dtype=np.int8)  # 0 = not visited, 1 = visited    
      
        # sources are considered visited
        self.visited_targets[self.source_mask] = True      
        
        # truck to act
        self.active_truck = 0
        
        # tours start with initial positions
        #self.tours = [[pos] for pos in self.truck_starts]

        # reset total_time_bytruck
        self.trucks_dict_state = {
            i: TruckState(total_time=0.0, tour=[self.truck_starts[i]])
            for i in range(self.num_trucks)
        }
        
        return self._get_obs(), {}

    def _get_obs(self):
        return  {
            "nodes": self.nodes.numpy(),
            "is_target": self.target_mask.astype(np.int8),
            "visited_targets": self.visited_targets.astype(np.int8),
            "current_trucks": self.truck_positions.copy(), #copy to avoid reference issues
        }

    def step(self, action):
        self.num_steps += 1
        truck_id = self.active_truck        
        terminated = False        
        prev_node = self.truck_positions[truck_id]
        reward = 0.0
        if action==self.num_nodes: # NO-OP action
            # Skip action, move to next     
            reward -=  100.0  # Heavy penalty for NO-OP to encourage visiting customers
            # print(f"Truck {truck_id} took NO-OP action. Penalizing heavily!!!!!!!!!!")
        else:
            reward += 10.0  # Reward for visiting a new target
            self.truck_positions[truck_id] = action
            #if (self.visited_targets[action] == True):
            #    print(f"Truck!!!!!!!!!!!!!!!!!!! {truck_id} visited an already visited target: {action}.")
            self.visited_targets[action] = True        
            self.trucks_dict_state[truck_id].tour.append(action)
            
            dist = self.time_matrix[prev_node, action]
            
            #if self.trucks_dict_state[truck_id].total_time + dist > 24.0:
            #    reward -= 100  # penalización fuerte
            #else:
            #    reward -= dist
            reward -= dist
            self.trucks_dict_state[truck_id].total_time += dist
        
            
        done = self.visited_targets[self.target_mask].all()

        # Buscar el siguiente camión disponible (que no supere 24h)
        self.active_truck, terminated = self._get_next_truck_id()  
        if (terminated):
            print(f"All trucks exceeded 24h xxxx. Terminating episode.")

        unvisited_count = (self.visited_targets == False).sum().item()
        #print(f"Unvisited targets count: {unvisited_count}")
        reward -= (unvisited_count * 500.0) # Heavy penalty # Goal: maximize clients

       
        if self.num_steps >= self.num_nodes+500:
            terminated = True
            reward -= 1000 # Heavy penalty for too many steps (to prevent infinite loops)
            #print(f"Rexedio de pasos  yyyy. Terminating episode.")
        return self._get_obs(), reward, done, terminated, {}
    
    
    def _get_next_truck_id(self):
        next_truck = (self.active_truck + 1) % self.num_trucks
        for _ in range(self.num_trucks):
            if self.trucks_dict_state[next_truck].total_time <= 24:
                return next_truck, False
            next_truck = (next_truck + 1) % self.num_trucks
        # Si todos los camiones superan 24h, devolver -1 y terminated
        return -1, True
