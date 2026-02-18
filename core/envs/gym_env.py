import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MDVRPGymEnv(gym.Env):
    def __init__(self, data, max_steps=150, max_daily_time=12.0):
        super().__init__()
        self.data = data
        self.max_steps = max_steps
        self.max_daily_time = max_daily_time

        self.node_features = data["node_features"]
        self.depots = data["depots"]
        self.customers = data["customers"]
        self.trucks = data["trucks"]
        self.num_nodes = data["num_nodes"]
        self.num_trucks = len(self.trucks)
        self.time_matrix = data["time_matrix"]
        self.truck_starts = [t.depot_idx for t in self.trucks]

        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self.num_nodes, self.node_features.shape[1]),
            dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([self.num_trucks, self.num_nodes])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.truck_positions = [start for start in self.truck_starts]
        self.truck_times = [0.0 for _ in range(self.num_trucks)]
        self.visited_customers = torch.zeros(self.num_nodes, dtype=torch.float32)
        self._tours = [[start] for start in self.truck_starts]
        self.current_step = 0
        return self.get_state(), self._get_info()

    def step(self, action):
        truck_id, next_node = action
        truck_id, next_node = int(truck_id), int(next_node)
        current_node = self.truck_positions[truck_id]
        travel_time = float(self.time_matrix[current_node, next_node])
        
        potential_time = self.truck_times[truck_id] + travel_time
        if potential_time > self.max_daily_time:
            return self.get_state(), -100.0, False, False, self._get_info()

        self.truck_times[truck_id] = potential_time
        self.truck_positions[truck_id] = next_node
        self._tours[truck_id].append(next_node)

        is_customer = next_node not in self.truck_starts
        first_visit = (is_customer and self.visited_customers[next_node] == 0)
        
        reward = -0.1 * travel_time 
        
        if is_customer:
            if first_visit:
                self.visited_customers[next_node] = 1.0
                reward += 1000.0 
                truck_cluster = truck_id 
                cluster_mask = (self.node_features[:, 4] == truck_cluster)
                is_customer_mask = torch.ones(self.num_nodes, dtype=torch.bool)
                for depot in self.truck_starts: is_customer_mask[depot] = False
                cluster_customers = cluster_mask & is_customer_mask
                if torch.all(self.visited_customers[cluster_customers] == 1.0):
                    reward += 2000.0 
            else:
                reward -= 10.0 
        
        self.current_step += 1
        
        all_trucks_done = all(t >= self.max_daily_time - 0.5 for t in self.truck_times)
        terminated = self._all_customers_visited() or all_trucks_done
        truncated = self.current_step >= self.max_steps
        
        if terminated or truncated:
            reward += self._compute_terminal_penalties()

        return self.get_state(), float(reward), terminated, truncated, self._get_info()

    

    def get_state(self):
        state = self.node_features.clone()
        state[:, 3] = self.visited_customers 
        return state

    def _get_info(self):
       
        return {
            "step": self.current_step,
            "visited_count": int(self.visited_customers.sum().item()),
            "total_truck_time": sum(self.truck_times)
        }

    def _compute_reward(self, travel_time, node, first_visit, time_violation, current_node):
        reward = -0.01 * travel_time
        reward += 500.0 if first_visit else -20.0
        if time_violation: reward -= 500.0
        
        # Cluster logic
        curr_c = int(self.node_features[current_node, 4].item())
        next_c = int(self.node_features[node, 4].item())
        reward += 20.0 if curr_c == next_c else -10.0
        
        return reward

    def _compute_terminal_penalties(self):
        bonus_or_penalty = 0.0
        
        for t_id in range(self.num_trucks):
            if self.truck_positions[t_id] in self.truck_starts:
                if len(self._tours[t_id]) > 1:
                    bonus_or_penalty += 500.0
            else:
                bonus_or_penalty -= 200.0
        for t_time in self.truck_times:
            time_saved = max(0, self.max_daily_time - t_time)
            bonus_or_penalty += (time_saved * 100.0)

        unvisited = (self.num_nodes - len(self.depots)) - self.visited_customers.sum().item()
        bonus_or_penalty -= unvisited * 1000.0 
        
        return bonus_or_penalty
       
    def _all_customers_visited(self):
        return self.visited_customers.sum().item() >= (self.num_nodes - len(self.depots))

    @property
    def tours(self):
        return self._tours

    def mask_actions(self):
        truck_mask = torch.zeros(self.num_trucks, dtype=torch.bool)
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        #Mask customers already visited
        node_mask[self.visited_customers == 1] = True

        #Mask trucks that are out of time
        for t_id in range(self.num_trucks):
            # if truck has less than 0.5h left, consider it done
            if self.truck_times[t_id] >= self.max_daily_time - 0.5:
                truck_mask[t_id] = True
                
        return truck_mask, node_mask