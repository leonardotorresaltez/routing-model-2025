import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loader_lib.data_loader import FleetStatus

from .policy import FactorizedFleetPolicy


# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:


        
    def __init__(self, cfg):
        self.cfg = cfg
 
        
        self.policy = FactorizedFleetPolicy(embed_dim=cfg.embed_dim, cfg=cfg, input_features_size=10)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for PPO
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []   # V(s_t) from critic
        self.terminal_bonus = 0.0  # fleet_time + coverage, redistributed undiscounted

        # PPO observation/action buffers (for re-evaluation across epochs)
        self.obs_features = []
        self.obs_truck_pos = []
        self.obs_visited = []
        self.obs_inactive = []
        self.obs_coords = []
        self.actions_truck = []
        self.actions_node = []

        # Episode-level cache for tensors that are constant within an episode
        self._ep_time_matrix = None    # [N, N] GPU — uploaded once per episode
        self._ep_coords = None         # [N, 2] GPU — constant per episode
        self._ep_is_target = None      # [N, 1] GPU — constant per episode
        self._ep_home_counts = None    # [N, 1] GPU — constant per episode
        self._ep_min_dist_depot = None # [N, 1] GPU — constant per episode
        self._ep_return_times = None   # [T, N] GPU — return times to home depot, constant per episode



    def act(self, obs):

        # --- Initialize episode cache on first step of each episode ---
        if len(self.log_probs) == 0:
            self._init_episode_cache(obs)

        # masking: Calculate valid moves
        visited_enriched_tensor = self._apply_time_constraints_v3(obs)
        
        # enrich observation space concatenating 
        observation_space_as_features = self._get_enriched_observation_space(obs)  
        
        # masking: inactive trucks   
        inactive_trucks_mask = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.bool).to(self.cfg.device)
        
        coords = torch.tensor(obs["nodes"], dtype=torch.float32, device=self.cfg.device)
        truck, node = self._select_action(
            observation_space_as_features,
            obs["truck_positions"],
            visited_enriched_tensor,
            inactive_trucks_mask,
            coords)

        # Store obs + action for PPO re-evaluation
        self.obs_features.append(observation_space_as_features.detach())
        self.obs_truck_pos.append(obs["truck_positions"].copy())
        self.obs_visited.append(visited_enriched_tensor.detach())
        self.obs_inactive.append(inactive_trucks_mask.detach())
        self.obs_coords.append(coords.detach())
        self.actions_truck.append(int(truck))
        self.actions_node.append(int(node))

        return int(truck), int(node)
        

    def store_reward(self, reward):
        if self.cfg.debug: print(f"DEBUG: Storing reward: {reward}")
        self.rewards.append(reward)

    def store_terminal_bonus(self, bonus):
        """Store fleet_time + coverage terminal reward for undiscounted redistribution."""
        self.terminal_bonus = bonus

    def update(self):
        """
        PPO: Proximal Policy Optimization.
        Prevents mode collapse by clipping π_new/π_old ∈ [1-ε, 1+ε], so no single
        gradient update can push the policy too far from its current distribution.
        Advantages and old log probs are fixed across ppo_epochs; only the policy
        parameters change, constrained by the clip.
        """
        n = len(self.log_probs)
        assert n == len(self.rewards), \
            f"MISALIGNMENT DETECTED! You have {n} actions but {len(self.rewards)} rewards."

        if self.cfg.debug: print(f"DEBUG: Log_Probs: {n} | Rewards: {len(self.rewards)}")

        # --- 1. Compute discounted returns G_t (backward) ---
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.cfg.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)

        # --- 1b. Redistribute terminal bonus to ALL steps (undiscounted) ---
        returns = returns + self.terminal_bonus
        self.terminal_bonus = 0.0

        # --- 2. Advantages fixed across epochs (computed from first-pass critic values) ---
        values_first = torch.stack(self.values).to(self.cfg.device)
        advantages = returns - values_first.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-9)

        # --- 3. Old log probs: reference point that the clip is measured against ---
        old_log_probs = torch.stack(self.log_probs).detach()  # [n]

        # --- 4. Pre-build batched tensors for fast PPO re-evaluation ---
        device = self.cfg.device
        features_batch  = torch.stack(self.obs_features)   # [n, N_nodes, F]
        visited_batch   = torch.stack(self.obs_visited)    # [n, T, N_nodes]
        inactive_batch  = torch.stack(self.obs_inactive)   # [n, T]
        coords_batch    = torch.stack(self.obs_coords)     # [n, N_nodes, 2]
        truck_pos_batch = torch.from_numpy(
            np.array(self.obs_truck_pos, dtype=np.int64)
        ).to(device)  # [n, T]
        actions_truck_t = self.actions_truck   # plain Python list[int]
        actions_node_t  = self.actions_node    # plain Python list[int]
        num_nodes = self.obs_features[0].shape[0]

        # --- 5. PPO epochs with mini-batch gradient accumulation ---
        # Chunk size limits the GNN intermediate tensor to ~chunk × E × D bytes.
        # E.g. chunk=32, E=20705, D=256 → 32×20705×256×4 ≈ 677 MB (fits in 8 GB VRAM).
        CHUNK = self.cfg.ppo_chunk_size

        for _ in range(self.cfg.ppo_epochs):
            self.optimizer.zero_grad()
            total_entropy = torch.tensor(0.0, device=device)

            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                chunk_n = end - start

                # Batched forward for this chunk only
                tp_b, np_b, vals_b = self.policy(
                    features_batch[start:end],
                    truck_pos_batch[start:end],
                    visited_batch[start:end],
                    inactive_batch[start:end],
                    coords_batch[start:end],
                )

                chunk_log_probs, chunk_entropies = [], []
                for j, i in enumerate(range(start, end)):
                    a_truck = actions_truck_t[i]
                    a_node  = actions_node_t[i]

                    truck_dist = torch.distributions.Categorical(tp_b[j])
                    row = np_b[j][a_truck]
                    if a_node == num_nodes or torch.allclose(row, torch.full_like(row, row[0].item())):
                        node_p = torch.zeros(num_nodes + 1, device=device)
                        node_p[-1] = 1.0
                    else:
                        node_p = row

                    node_dist = torch.distributions.Categorical(node_p)
                    a_truck_t2 = torch.tensor(a_truck, device=device)
                    a_node_t2  = torch.tensor(a_node,  device=device)
                    chunk_log_probs.append(truck_dist.log_prob(a_truck_t2) + node_dist.log_prob(a_node_t2))
                    chunk_entropies.append(truck_dist.entropy() + node_dist.entropy())

                chunk_lp = torch.stack(chunk_log_probs)
                entropy_chunk = torch.stack(chunk_entropies).mean()
                total_entropy = total_entropy + entropy_chunk * (chunk_n / n)

                ratio = torch.exp(chunk_lp - old_log_probs[start:end])
                clipped = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip)
                policy_loss = -torch.min(ratio * advantages[start:end],
                                         clipped * advantages[start:end]).mean()
                value_loss  = F.mse_loss(vals_b, returns_norm[start:end])
                chunk_loss  = (policy_loss
                               + self.cfg.value_coef * value_loss
                               - self.cfg.entropy_bonus * entropy_chunk) * (chunk_n / n)
                chunk_loss.backward()  # accumulates gradients across chunks

            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        loss        = chunk_loss      # last chunk loss (for logging)
        entropy_loss = total_entropy  # weighted entropy (for logging)

        # --- 5. Clear all buffers ---
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.values.clear()
        self.obs_features.clear()
        self.obs_truck_pos.clear()
        self.obs_visited.clear()
        self.obs_inactive.clear()
        self.obs_coords.clear()
        self.actions_truck.clear()
        self.actions_node.clear()

        return loss.item(), entropy_loss.item(), grad_norm.item(), returns.mean().item()
      
    
    
    def _init_episode_cache(self, obs):
        """Upload and compute all tensors that are constant for the entire episode."""
        device = self.cfg.device
        tm = obs["time_matrix"]
        # Upload time_matrix once (it never changes within an episode)
        if isinstance(tm, torch.Tensor):
            self._ep_time_matrix = tm.to(device, dtype=torch.float32, non_blocking=True)
        else:
            self._ep_time_matrix = torch.tensor(tm, dtype=torch.float32, device=device)

        self._ep_coords = torch.tensor(obs["nodes"], dtype=torch.float32, device=device)
        self._ep_is_target = torch.tensor(obs["is_target"], dtype=torch.float32, device=device).unsqueeze(1)

        # home_counts: how many trucks start at each node
        num_nodes = self._ep_coords.shape[0]
        home_counts = torch.zeros(num_nodes, 1, device=device)
        for idx in obs["truck_starts"]:
            home_counts[idx] += 1
        self._ep_home_counts = home_counts

        # min_dist_to_depot: closest depot distance per node
        depot_indices = list(set(obs["truck_starts"]))
        dist_to_depots = self._ep_time_matrix[:, depot_indices]
        self._ep_min_dist_depot, _ = torch.min(dist_to_depots, dim=1, keepdim=True)

        # return_times[t, n] = time_matrix[n, truck_starts[t]] — for constraint mask
        truck_starts_t = torch.tensor(obs["truck_starts"], dtype=torch.long, device=device)
        self._ep_return_times = self._ep_time_matrix[:, truck_starts_t].T  # [T, N]

    def _get_enriched_observation_space(self, obs):
        """
        Concatenate spatial, status, and fleet context into a fixed-size feature vector.
        Invariant features (coords, is_target, home_counts, min_dist_depot) come from
        the episode cache — computed once, reused every step.
        """
        device = self.cfg.device
        num_nodes = self._ep_coords.shape[0]

        # Dynamic per-step features
        visited = torch.tensor(obs["visited_targets"], dtype=torch.float32, device=device).unsqueeze(1)
        inactive_mask = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.float32, device=device)
        truck_times = torch.tensor(obs["truck_times"], dtype=torch.float32, device=device)

        active_ratio  = (1.0 - inactive_mask).mean().reshape(1, 1)
        avg_fleet_time = truck_times.mean().reshape(1, 1)
        max_fleet_time = truck_times.max().reshape(1, 1)
        fleet_stats = torch.cat([active_ratio, avg_fleet_time, max_fleet_time], dim=1).expand(num_nodes, -1)

        # Min distance from any active truck to each node
        inactive_mask_bool = obs["inactive_trucks_mask"].astype(bool)
        active_positions = [i for i, inact in enumerate(inactive_mask_bool) if not inact]
        if active_positions:
            active_pos_t = torch.tensor(active_positions, dtype=torch.long, device=device)
            truck_positions_t = torch.tensor(obs["truck_positions"], dtype=torch.long, device=device)
            active_node_ids = truck_positions_t[active_pos_t]
            truck_dists = self._ep_time_matrix[active_node_ids]         # [active_T, N]
            min_dist_from_trucks = truck_dists.min(dim=0).values.unsqueeze(1)  # [N, 1]
        else:
            min_dist_from_trucks = torch.zeros(num_nodes, 1, device=device)

        # [Coords(2), Target(1), Visited(1), Fleet(3), Home(1), MinDepot(1), MinTruck(1)] = 10
        return torch.cat([
            self._ep_coords,
            self._ep_is_target,
            visited,
            fleet_stats,
            self._ep_home_counts,
            self._ep_min_dist_depot,
            min_dist_from_trucks,
        ], dim=1)


    # def _get_enriched_observation_space(self, obs):
    #     """
    #     Concatenate all observation space elements with dimension N into a single tensor.
    #     """
    #     is_target = torch.tensor(obs["is_target"], dtype=torch.float32).unsqueeze(1).to(self.cfg.device)  # Ensure Shape: (N, 1)
    #     visited_targets = torch.tensor(obs["visited_targets"], dtype=torch.float32).unsqueeze(1).to(self.cfg.device)  # Ensure Shape: (N, 1)
    #     time_matrix = torch.tensor(obs["time_matrix"], dtype=torch.float32).to(self.cfg.device)  # Shape: (N, N)

    #     # Concatenate all tensors with dimension N along the last axis
    #     enriched_tensor = torch.cat([time_matrix,is_target, visited_targets], dim=1)  # Shape: (N, N+2) if time_matrix is (N, N) and the others are (N, 1)

    #     return enriched_tensor
    
 
    
    
    def _apply_time_constraints_v3(self, obs):
        """
        Vectorized time-constraint masking — no Python loop over trucks.
        Uses cached time_matrix and return_times from _init_episode_cache.
        """
        device = self.cfg.device
        visited_mask = torch.tensor(obs["visited_targets"], dtype=torch.bool, device=device)
        # [T, N] — each row is the visited mask per truck
        masks = visited_mask.unsqueeze(0).expand(len(obs["truck_positions"]), -1).clone()

        truck_positions_t = torch.tensor(obs["truck_positions"], dtype=torch.long, device=device)  # [T]
        truck_times_t     = torch.tensor(obs["truck_times"],     dtype=torch.float32, device=device)  # [T]

        # travel_times[t, n] = time_matrix[truck_pos[t], n]
        travel_times = self._ep_time_matrix[truck_positions_t]  # [T, N]

        # total_times[t, n] = current_time[t] + travel[t,n] + return[t,n]
        total_times = truck_times_t.unsqueeze(1) + travel_times + self._ep_return_times  # [T, N]

        masks |= total_times > self.cfg.max_daily_delivery_time_each_truck
        return masks
        
     
                
        
    def _evaluate_action_log_prob(self, features, truck_pos, visited_mask, inactive_mask, coords, a_truck, a_node):
        """
        Re-evaluate log_prob and entropy for a stored (obs, action) pair under the current policy.
        Mirrors the NO-OP detection logic of _select_action so both paths are consistent.
        """
        truck_probs, node_probs, value = self.policy(features, truck_pos, visited_mask, inactive_mask, coords)

        truck_dist = torch.distributions.Categorical(truck_probs)
        num_nodes = features.shape[0]

        # Replicate NO-OP detection: if all node scores are uniform (all masked) → NO-OP was forced
        row = node_probs[a_truck]
        if a_node == num_nodes or torch.allclose(row, torch.full_like(row, row[0].item())):
            node_p = torch.zeros(num_nodes + 1, device=self.cfg.device)
            node_p[-1] = 1.0  # NO-OP has probability 1
        else:
            node_p = row

        node_dist = torch.distributions.Categorical(node_p)

        a_truck_t = torch.tensor(a_truck, device=self.cfg.device)
        a_node_t = torch.tensor(a_node, device=self.cfg.device)
        log_prob = truck_dist.log_prob(a_truck_t) + node_dist.log_prob(a_node_t)
        entropy = truck_dist.entropy() + node_dist.entropy()
        return log_prob, entropy, value

    def _select_action(self, nodes, truck_positions, visited_enriched, inactive_trucks_mask, coords):
        """
        Helper to select the next action or return NO-OP if all nodes are visited.
        """

        # Pass the mask to the policy
        truck_probs, node_probs, value = self.policy(nodes, truck_positions, visited_enriched, inactive_trucks_mask,coords)
        self.values.append(value)
        
        # ---- sample truck ----
        truck_dist = torch.distributions.Categorical(truck_probs)
        truck = truck_dist.sample()

        # ---- sample node for that truck ----
        num_nodes = nodes.shape[0]
        
        if torch.allclose(node_probs[truck], torch.full_like(node_probs[truck], node_probs[truck][0].item())):  # Check if all node probabilities for the selected truck are masked (== -1e9)
            node_probs = torch.zeros(num_nodes + 1, device=nodes.device)  # Create a new tensor for node probabilities
            node_probs[-1] = 1.0  # Assign probability 1 to the NO-OP action at the last index wich is -1 == num_nodes == NO-OP action
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