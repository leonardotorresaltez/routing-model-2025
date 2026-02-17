import torch

class MDVRPGymEnv:
    def __init__(self, data, max_steps=1000, max_daily_time=24.0):
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

        self.reset()

    def reset(self):
        self.truck_positions = [start for start in self.truck_starts]
        self.truck_times = [0.0 for _ in range(self.num_trucks)]
        self.visited_customers = torch.zeros(self.num_nodes, dtype=torch.float32)
        self._tours = [[start] for start in self.truck_starts]
        self.current_step = 0
        return self.get_state()

    @property
    def tours(self):
        return self._tours

    def get_state(self):
        visited_mask = self.visited_customers.unsqueeze(1)

        # node_features already contains: [lat, lon, demand, visited, cluster_id]
        state = self.node_features.clone()
        state[:, 3] = visited_mask.squeeze(1) 

        return state
    def _all_customers_visited(self):
        num_depots = len(self.depots)
        total_customers = self.num_nodes - num_depots
        return self.visited_customers.sum().item() >= total_customers

    def _is_done(self, time_violation):
        if self._all_customers_visited():
            return True
        if time_violation:
            return True
        if self.current_step >= self.max_steps:
            return True
        return False

    def _compute_reward(self, travel_time, node, first_visit, time_violation):
        reward = 0.0

        # small time penalty
        reward -= 0.01 * travel_time

        # customer visit reward

        if first_visit:
            reward += 500.0
        else:
            reward -= 20.0
        if time_violation:
            reward -= 500.0

        return reward

    def step(self, action):
        truck_id, next_node = action
        truck_id = int(truck_id)
        next_node = int(next_node)

        current_node = self.truck_positions[truck_id]

        travel_time = float(self.time_matrix[current_node, next_node])
        self.truck_times[truck_id] += travel_time

        self.truck_positions[truck_id] = next_node
        self._tours[truck_id].append(next_node)

        is_customer = next_node not in self.truck_starts

        first_visit = False
        if is_customer and self.visited_customers[next_node] == 0:
            first_visit = True

        time_violation = self.truck_times[truck_id] > self.max_daily_time

        reward = self._compute_reward(travel_time, next_node, first_visit, time_violation)

   
        current_cluster = int(self.node_features[current_node, 4].item())
        next_cluster = int(self.node_features[next_node, 4].item())

        if current_cluster == next_cluster:
            reward += 20.0
        else:
            reward -= 10.0

        if is_customer:
            self.visited_customers[next_node] = 1.0

        if len(self._tours[truck_id]) == 2 and is_customer:
            reward += 100.0

        for other_t in range(self.num_trucks):
            if other_t != truck_id and next_node in self._tours[other_t]:
                reward -= 20.0

        if self._all_customers_visited():
            reward += 5000.0

        self.current_step += 1
        done = self._is_done(time_violation)

        if done:
            for t_id in range(self.num_trucks):
                if len(self._tours[t_id]) == 1:
                    reward -= 300.0

            num_depots = len(self.depots)
            total_customers = self.num_nodes - num_depots
            visited = int(self.visited_customers.sum().item())
            unvisited = total_customers - visited
            reward -= unvisited * 50.0

        return self.get_state(), reward, done, {}

    def mask_actions(self):
        truck_mask = torch.zeros(self.num_trucks, dtype=torch.bool)
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        visited = self.visited_customers == 1
        node_mask[visited] = True

        # prevent depot revisits
        for t_id, depot_idx in enumerate(self.truck_starts):
            if self.truck_positions[t_id] != depot_idx:
                node_mask[depot_idx] = True

        return truck_mask, node_mask