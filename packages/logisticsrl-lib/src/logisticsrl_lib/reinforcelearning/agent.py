import torch
import torch.nn.functional as F
import torch.optim as optim

from .policy import FactorizedFleetPolicy


# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:


        
    def __init__(self, cfg, edge_index=None):
        self.cfg = cfg


        self.policy = FactorizedFleetPolicy(embed_dim=cfg.embed_dim, cfg=cfg, input_features_size=10, edge_index=edge_index)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.episodes, eta_min=1e-5)

        # Buffers for A2C
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []   # V(s_t) from critic
        self.terminal_bonus = 0.0  # fleet_time + coverage, redistributed undiscounted

        # Episode-level cache — invariants uploaded once per episode
        self._ep_time_matrix = None   # [N, N] GPU
        self._ep_coords = None        # [N, 2] GPU
        self._ep_is_target = None     # [N, 1] GPU
        self._ep_home_counts = None   # [N, 1] GPU
        self._ep_min_dist_depot = None  # [N, 1] GPU



    def _init_episode_cache(self, obs):
        device = self.cfg.device
        self._ep_time_matrix = torch.tensor(obs["time_matrix"], dtype=torch.float32, device=device)
        self._ep_coords = torch.tensor(obs["nodes"], dtype=torch.float32, device=device)
        self._ep_is_target = torch.tensor(obs["is_target"], dtype=torch.float32, device=device).unsqueeze(1)
        num_nodes = self._ep_coords.shape[0]
        truck_starts = obs["truck_starts"]
        home_counts = torch.zeros(num_nodes, 1, device=device)
        for idx in truck_starts:
            home_counts[idx] += 1
        self._ep_home_counts = home_counts
        depot_indices = list(set(truck_starts))
        dist_to_depots = self._ep_time_matrix[:, depot_indices]
        self._ep_min_dist_depot, _ = torch.min(dist_to_depots, dim=1, keepdim=True)
        # [T, N]: return time from each node to each truck's home depot
        truck_starts_t = torch.tensor(truck_starts, dtype=torch.long, device=device)
        self._ep_return_times = self._ep_time_matrix[:, truck_starts_t].T  # [T, N]

    def act(self, obs):
        # Init cache at the start of each episode (first act call)
        if len(self.log_probs) == 0:
            self._init_episode_cache(obs)

        device = self.cfg.device
        # Pre-convert moving parts for this step just once to reduce CPU-GPU sync jitter
        visited_t = torch.tensor(obs["visited_targets"], dtype=torch.float32, device=device)
        visited_bool_t = visited_t.bool()
        inactive_bool_t = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.bool, device=device)
        truck_positions_t = torch.tensor(obs["truck_positions"], dtype=torch.long, device=device)
        truck_times_t = torch.tensor(obs["truck_times"], dtype=torch.float32, device=device)

        # masking: Calculate valid moves
        visited_enriched_tensor = self._apply_time_constraints_v3(visited_bool_t, truck_positions_t, truck_times_t)
        
        # enrich observation space concatenating 
        observation_space_as_features = self._get_enriched_observation_space(
            visited_t, inactive_bool_t, truck_times_t, truck_positions_t
        )  
        
        truck, node = self._select_action(
            observation_space_as_features,
            truck_positions_t,
            visited_enriched_tensor,
            inactive_bool_t)

        return int(truck), int(node)
        

    def store_reward(self, reward):
        if self.cfg.debug: print(f"DEBUG: Storing reward: {reward}")
        self.rewards.append(reward)

    def store_terminal_bonus(self, bonus):
        """Store fleet_time + coverage terminal reward for undiscounted redistribution."""
        self.terminal_bonus = bonus

    def update(self):
        """
        A2C: Actor-Critic policy gradient.
        Policy loss uses advantages A_t = G_t - V(s_t) instead of raw returns.
        Value loss minimizes MSE between V(s_t) and G_t.
        """
        n_probs = len(self.log_probs)
        n_rewards = len(self.rewards)

        if self.cfg.debug: print(f"DEBUG: Log_Probs: {n_probs} | Rewards: {n_rewards}")

        assert n_probs == n_rewards, \
            f"MISALIGNMENT DETECTED! You have {n_probs} actions but {n_rewards} rewards."


        # Apply terminal bonus to last reward
        if len(self.rewards) > 0:
            self.rewards[-1] += self.terminal_bonus
        self.terminal_bonus = 0.0

        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.cfg.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        
        
        # Stack stored tensors 
        values = torch.stack(self.values).to(self.cfg.device)    # [T]
        log_probs = torch.stack(self.log_probs).to(self.cfg.device)        
        entropy = torch.stack(self.entropies).to(self.cfg.device)
        
        
        #advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        

        # losses
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        #Total loss ---
        loss = (actor_loss
                + self.cfg.value_coef * critic_loss
                + self.cfg.entropy_bonus * entropy_loss)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.values.clear()

        # returns.mean() = mean discounted return per episode (increases as agent improves).
        # Previously logged advantages.mean() which is always ~0 by construction (useless).
        return loss.item(), entropy_loss.item(), grad_norm.item(), returns.mean().item()
      
    
    
    def _get_enriched_observation_space(self, visited_t, inactive_mask_bool, truck_times_t, truck_positions_t):
        """
        Concatenate spatial, status, and fleet context into a fixed-size feature vector.
        Invariants (coords, is_target, home_counts, min_dist_depot) are read from episode cache.
        """
        device = self.cfg.device
        num_nodes = self._ep_coords.shape[0]

        # 1. Node status — changes each step
        visited = visited_t.unsqueeze(1)

        # 2. Global Fleet Context (Broadcasted) — changes each step
        inactive_mask_f = inactive_mask_bool.float()
        active_ratio = (1.0 - inactive_mask_f).mean().reshape(1, 1)
        avg_fleet_time = truck_times_t.mean().reshape(1, 1)
        max_fleet_time = truck_times_t.max().reshape(1, 1)
        fleet_stats = torch.cat([active_ratio, avg_fleet_time, max_fleet_time], dim=1).repeat(num_nodes, 1)

        # 3. Min distance from any active truck to each node — changes each step
        active_idx = (~inactive_mask_bool).nonzero(as_tuple=True)[0]
        if active_idx.numel() > 0:
            active_truck_positions = truck_positions_t[active_idx]
            truck_dists = self._ep_time_matrix[active_truck_positions, :]  # [active_T, N]
            min_dist_from_trucks, _ = torch.min(truck_dists, dim=0, keepdim=True)
            min_dist_from_trucks = min_dist_from_trucks.T  # [N, 1]
        else:
            min_dist_from_trucks = torch.zeros(num_nodes, 1, device=device)

        # 4. Final Concatenation — Total Dimension: 10
        # [Coords(2), Target(1), Visited(1), Fleet(3), Home(1), MinDepot(1), MinTruck(1)]
        return torch.cat([
            self._ep_coords,
            self._ep_is_target,
            visited,
            fleet_stats,
            self._ep_home_counts,
            self._ep_min_dist_depot,
            min_dist_from_trucks
        ], dim=1)




    
    def _apply_time_constraints_v3(self, visited_mask_bool, truck_positions_t, truck_times_t):
        """
        Vectorized time constraint masking — no Python loop over trucks.
        Uses episode-cached time_matrix and return_times.
        """
        device = self.cfg.device
        masks = visited_mask_bool.unsqueeze(0).expand(len(truck_positions_t), -1).clone()  # [T, N]

        travel_times = self._ep_time_matrix[truck_positions_t]   # [T, N]
        total_times = truck_times_t.unsqueeze(1) + travel_times + self._ep_return_times  # [T, N]
        masks |= total_times > self.cfg.max_daily_delivery_time_each_truck

        return masks        
        
     
                
        
    def _select_action(self, nodes, truck_positions, visited_enriched, inactive_trucks_mask):
        """
        Helper to select the next action or return NO-OP if all nodes are visited.
        """

        # Pass the mask to the policy
        truck_probs, node_probs, value = self.policy(nodes, truck_positions, visited_enriched, inactive_trucks_mask)
        self.values.append(value)

        # ---- sample truck ----
        truck_dist = torch.distributions.Categorical(truck_probs)
        truck = truck_dist.sample()

        # ---- sample node for that truck ----
        num_nodes = nodes.shape[0]

        if visited_enriched[truck].all():  # all nodes masked for this truck → NO-OP
            node_probs = torch.zeros(num_nodes + 1, device=nodes.device)
            node_probs[-1] = 1.0
        else:
            node_probs = node_probs[truck]
        
        node_dist = torch.distributions.Categorical(node_probs)
        node = node_dist.sample()

        # ---- log prob joint ----
        log_prob = truck_dist.log_prob(truck) + node_dist.log_prob(node)
        self.log_probs.append(log_prob)
        entropy = truck_dist.entropy() + node_dist.entropy()
        self.entropies.append(entropy)
        return truck.item(), node.item()