import torch
import gymnasium as gym
import numpy as np
class MDVRPGymEnv(gym.Env):
    def __init__(self, data, max_steps=400, max_daily_time=24.0):
        super().__init__()
        self.data = data
        self.max_steps = max_steps
        self.max_daily_time = max_daily_time
        self.trucks = data["trucks"] 
        self.num_trucks = len(self.trucks)
        self.num_nodes = data["num_nodes"]
        self.truck_starts = [t.depot_idx for t in self.trucks]
        self.depot_indices = torch.tensor(data["depot_indices"])
        self.cluster_ids = self.data["cluster_ids"] 
        #Observation Space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_nodes, 5), 
            dtype=np.float32
        )
        #Action Space
        self.action_space = gym.spaces.MultiDiscrete(
            [self.num_nodes] * self.num_trucks
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.truck_positions = self.truck_starts.copy()
        self.truck_times = [0.0] * self.num_trucks
        self.visited_customers = torch.zeros(self.num_nodes)
        self._tours = [[s] for s in self.truck_starts]
        self.current_step = 0
        return self.get_state(), {}

    def get_state(self):
        state = self.data["node_features"].clone()
        state[:, 3] = self.visited_customers
        return state

    def step(self, actions):
        total_reward = 0.0
        tick_claims = {}

        for t_id in range(self.num_trucks):
            next_node = int(actions[t_id])
            curr_node = self.truck_positions[t_id]
            home = self.truck_starts[t_id]
            truck_obj = self.trucks[t_id]
          
            # 1. Skip trucks that are already back home and finished
            if curr_node == home and len(self._tours[t_id]) > 1:
                continue

            # 2. PHYSICAL FEASIBILITY CHECK (The 24h Guard)
            # Calculate time to reach next node + time to get back home from there
            t_to_next = float(self.data["time_matrix"][curr_node, next_node])
            t_from_next_to_home = float(self.data["time_matrix"][next_node, home])
            
            # If proposed move violates 24h limit, force them home instead
            if self.truck_times[t_id] + t_to_next + t_from_next_to_home > self.max_daily_time:
                if curr_node != home:
                    next_node = home
                    time_cost = float(self.data["time_matrix"][curr_node, home])
                else:
                    # Already home, just stay there
                    continue 
            else:
                time_cost = t_to_next

            # 3. Update Truck State
            self.truck_times[t_id] += time_cost
            self.truck_positions[t_id] = next_node
            self._tours[t_id].append(next_node)

            # 4. Reward Logic
            is_depot = next_node in self.depot_indices
            if not is_depot:
                if self.visited_customers[next_node] == 0:
                    self.visited_customers[next_node] = 1.0
                    delivery_reward = 500.0
                    dist_from_depot = float(self.data["time_matrix"][home, next_node])
                    delivery_reward += (dist_from_depot * 5.0)
                    
                    if int(self.cluster_ids[next_node]) == truck_obj.target_cluster:
                        delivery_reward += 50.0
                    total_reward += delivery_reward
                else:
                    total_reward -= 100.0 # Re-visit penalty
                
                if next_node in tick_claims:
                    total_reward -= 50.0 # Collision penalty
                tick_claims[next_node] = t_id
            
            # Efficiency Penalty
            total_reward -= 0.1 * time_cost

        self.current_step += 1
        
        # 5. Termination check
        total_customers = self.num_nodes - len(self.depot_indices)
        all_visited = self.visited_customers.sum() >= total_customers
        all_home = all([self.truck_positions[i] == self.truck_starts[i] and len(self._tours[i]) > 1 
                        for i in range(self.num_trucks)])
        
        terminated = bool(all_visited or all_home)
        truncated = bool(self.current_step >= self.max_steps)
        
        if all_visited:
            total_reward += 4000.0 

        return self.get_state(), float(total_reward), terminated, truncated, {}

    def mask_actions(self):
        n_mask = torch.zeros((self.num_trucks, self.num_nodes), dtype=torch.bool)
        t_mask = torch.zeros(self.num_trucks, dtype=torch.bool)
        
        for i in range(self.num_trucks):
            home = self.truck_starts[i]
            curr_pos = self.truck_positions[i]
            curr_time = self.truck_times[i]
            if curr_pos == home and len(self._tours[i]) > 1:
                n_mask[i, :] = True
                n_mask[i, home] = False
                t_mask[i] = True
                continue
            n_mask[i, self.visited_customers == 1] = True
            n_mask[i, curr_pos] = True

            for n in range(self.num_nodes):
                if n_mask[i, n] or n == home:
                    continue
                
                t_to = float(self.data["time_matrix"][curr_pos, n])
                t_back = float(self.data["time_matrix"][n, home])
                
                if curr_time + t_to + t_back > self.max_daily_time:
                    n_mask[i, n] = True

            reachable = (~n_mask[i]).clone()
            reachable[self.depot_indices] = False
            
            if not reachable.any():
                n_mask[i, :] = True
                n_mask[i, home] = False 
                t_mask[i] = True 
                
        return t_mask, n_mask

    @property
    def tours(self):
        return self._tours