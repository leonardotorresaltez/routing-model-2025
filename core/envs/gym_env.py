import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MDVRPGymEnv(gym.Env):
    def __init__(self, data, max_steps=1000, max_daily_time=24.0):
        super().__init__() # Initialize the parent class
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

        # Fix: Box bounds should not be infinity for stable training
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self.num_nodes, self.node_features.shape[1]),
            dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([self.num_trucks, self.num_nodes])

    def reset(self, seed=None, options=None):
        # Gymnasium reset logic
        super().reset(seed=seed)
        
        self.truck_positions = [start for start in self.truck_starts]
        self.truck_times = [0.0 for _ in range(self.num_trucks)]
        self.visited_customers = torch.zeros(self.num_nodes, dtype=torch.float32)
        self._tours = [[start] for start in self.truck_starts]
        self.current_step = 0
        
        # FIX: Return (state, info)
        return self.get_state(), self._get_info()

    def step(self, action):
        truck_id, next_node = action
        truck_id = int(truck_id)
        next_node = int(next_node)

        current_node = self.truck_positions[truck_id]
        travel_time = float(self.time_matrix[current_node, next_node])
        
        # Update State
        self.truck_times[truck_id] += travel_time
        self.truck_positions[truck_id] = next_node
        self._tours[truck_id].append(next_node)

        # Logic Checks
        is_customer = next_node not in self.truck_starts
        first_visit = (is_customer and self.visited_customers[next_node] == 0)
        
        if is_customer:
            self.visited_customers[next_node] = 1.0

        time_violation = self.truck_times[truck_id] > self.max_daily_time

        # Rewards
        reward = self._compute_reward(travel_time, next_node, first_visit, time_violation, current_node)

        self.current_step += 1
        
        # Gymnasium Logic: Separate Terminal from Truncated
        terminated = self._all_customers_visited() or time_violation
        truncated = self.current_step >= self.max_steps
        
        # Add terminal logic rewards
        if terminated or truncated:
            reward += self._compute_terminal_penalties()

        # FIX: Return 5 values: (obs, reward, terminated, truncated, info)
        return self.get_state(), float(reward), terminated, truncated, self._get_info()

    def get_state(self):
        state = self.node_features.clone()
        state[:, 3] = self.visited_customers 
        return state

    def _get_info(self):
        # Helpful for debugging/logging
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
        penalty = 0.0
        # Penalty for empty trucks
        for t_id in range(self.num_trucks):
            if len(self._tours[t_id]) == 1: penalty -= 300.0
        
        # Penalty for unvisited customers
        unvisited = (self.num_nodes - len(self.depots)) - self.visited_customers.sum().item()
        penalty -= unvisited * 50.0
        return penalty

    def _all_customers_visited(self):
        return self.visited_customers.sum().item() >= (self.num_nodes - len(self.depots))

    @property
    def tours(self):
        return self._tours

    def mask_actions(self):
        truck_mask = torch.zeros(self.num_trucks, dtype=torch.bool)
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        node_mask[self.visited_customers == 1] = True
        for t_id, depot_idx in enumerate(self.truck_starts):
            if self.truck_positions[t_id] != depot_idx:
                node_mask[depot_idx] = True
        return truck_mask, node_mask