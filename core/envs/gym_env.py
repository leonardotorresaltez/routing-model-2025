import gymnasium as gym
import numpy as np
import torch


class MDVRPGymEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, data, max_steps=200, max_daily_time=24.0):
        super().__init__()

        self.data = data
        self.max_steps = max_steps
        self.max_daily_time = max_daily_time

        # Core problem data
        self.trucks = data["trucks"]
        self.num_trucks = len(self.trucks)
        self.num_nodes = data["num_nodes"]

        self.time_matrix = data["time_matrix"].float()
        self.truck_starts = torch.tensor([t.depot_idx for t in self.trucks])
        self.depot_indices = torch.tensor(data.get("depot_indices", [0]))

        if len(self.depot_indices) == 0:
            self.depot_indices = torch.tensor([0])

        # Customers = all non-depot nodes
        self.customer_mask = torch.ones(self.num_nodes, dtype=torch.bool)
        self.customer_mask[self.depot_indices] = False
        self.total_customers = int(self.customer_mask.sum())

        # Observation space: node features (you already have this)
        self.feat_dim = data["node_features"].shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes, self.feat_dim),
            dtype=np.float32,
        )

        # Action space: one node per truck
        self.action_space = gym.spaces.MultiDiscrete(
            [self.num_nodes] * self.num_trucks
        )

        # --- Reward scaling based on time matrix ---
        tm = self.time_matrix
        mask = tm > 0
        self.time_mean = tm[mask].mean().item()
        self.time_std = tm[mask].std().item() if tm[mask].numel() > 1 else 1.0

        # Reward coefficients (tune these)
        self.r_visit = 2.0          # reward for visiting a new customer
        self.r_collision = -2.0     # penalty for trying to visit an already served / collided node
        self.r_unvisited = -5.0     # penalty per unvisited customer at the end
        self.r_finish_bonus = 5.0   # bonus for serving all customers
        self.r_time_scale = 0.5     # weight for travel-time penalty

        self.reset()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.truck_positions = self.truck_starts.clone().tolist()
        self.truck_times = [0.0] * self.num_trucks
        self.visited_customers = torch.zeros(self.num_nodes, dtype=torch.float32)
        self._tours = [[p] for p in self.truck_positions]
        self.current_step = 0

        obs = self.get_state()
        return obs, {}

    def get_state(self):
        state = self.data["node_features"].clone()
        # Assume column 3 is "visited" flag
        if state.shape[1] > 3:
            state[:, 3] = self.visited_customers
        return state

    def step(self, actions):
        """
        actions: tensor/list of length num_trucks
        """
        total_reward = 0.0
        visited_this_step = set()

        for t in range(self.num_trucks):
            next_node = int(actions[t])
            curr = int(self.truck_positions[t])
            home = int(self.truck_starts[t])
            is_depot = next_node in self.depot_indices.tolist()

            # --- Collision / double-visit prevention ---
            if not is_depot and (
                self.visited_customers[next_node] == 1
                or next_node in visited_this_step
            ):
                # Truck "wastes" its decision
                total_reward += self.r_collision
                continue

            # Travel
            t_to_next = float(self.time_matrix[curr, next_node])
            self.truck_times[t] += t_to_next
            self.truck_positions[t] = next_node
            self._tours[t].append(next_node)

            # Reward for visiting a new customer
            if not is_depot and self.visited_customers[next_node] == 0:
                self.visited_customers[next_node] = 1
                visited_this_step.add(next_node)
                total_reward += self.r_visit

            # Time / distance penalty (normalized-ish)
            # Larger travel times => more negative
            norm_time = (t_to_next - self.time_mean) / (self.time_std + 1e-6)
            total_reward -= self.r_time_scale * norm_time

        self.current_step += 1

        # --- Termination logic ---
        num_visited = int(self.visited_customers[self.customer_mask].sum())
        all_delivered = (num_visited >= self.total_customers)

        terminated = bool(all_delivered)
        truncated = self.current_step >= self.max_steps

        # --- Final rewards at episode end ---
        if terminated:
            # Bonus for completing all deliveries
            total_reward += self.r_finish_bonus

            # Mild efficiency shaping: penalize excessive total fleet time
            total_fleet_time = sum(self.truck_times)
            expected_time = self.time_mean * self.total_customers
            norm_fleet = (total_fleet_time - expected_time) / (self.time_std + 1e-6)
            total_reward -= 0.1 * norm_fleet

        elif truncated:
            # Penalize remaining unvisited customers
            unvisited = self.total_customers - num_visited
            total_reward += self.r_unvisited * unvisited

        obs = self.get_state()
        return obs, float(total_reward), terminated, truncated, {}

    # ------------------------------------------------------------------ #
    # Action masking
    # ------------------------------------------------------------------ #

    def mask_actions(self):
        """
        Returns:
            t_mask: (num_trucks,) bool tensor (not really used here, kept for compatibility)
            n_mask: (num_trucks, num_nodes) bool tensor, True = action not allowed
        """
        n_mask = torch.zeros((self.num_trucks, self.num_nodes), dtype=torch.bool)
        t_mask = torch.zeros(self.num_trucks, dtype=torch.bool)

        for t in range(self.num_trucks):
            curr = int(self.truck_positions[t])
            home = int(self.truck_starts[t])

            # 1. Mask already visited customers
            n_mask[t, self.visited_customers == 1] = True

            # 2. Time feasibility: must be able to go to node and back to home within max_daily_time
            t_to_all = self.time_matrix[curr]          # (num_nodes,)
            t_back_all = self.time_matrix[:, home]     # (num_nodes,)
            predicted_arrival = self.truck_times[t] + t_to_all + t_back_all
            n_mask[t, predicted_arrival > self.max_daily_time] = True

            # 3. Always allow depots as safe options
            n_mask[t, self.depot_indices] = False

            # 4. No self-loop on customers (can stay at depot if you want)
            if curr not in self.depot_indices.tolist():
                n_mask[t, curr] = True

        return t_mask, n_mask

    # ------------------------------------------------------------------ #
    # Tours property
    # ------------------------------------------------------------------ #

    @property
    def tours(self):
        return self._tours