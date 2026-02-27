import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from loader_lib.data_loader import FleetStatus
from .policy import FactorizedFleetPolicy

# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:

    _SUM_OTHER_DIM = 2
        
    def __init__(self, cfg, fleetStatus: FleetStatus):
        self.cfg = cfg
        self.fleetStatus = fleetStatus
        
        observation_space_as_features_dimenasion = fleetStatus.num_nodes() + self._SUM_OTHER_DIM # features dimension , check _get_enriched_observation_space
        self.policy = FactorizedFleetPolicy(size_dim=observation_space_as_features_dimenasion, embed_dim=cfg.embed_dim, cfg=cfg)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []
        
        self.entropies = []

        self.step_count = 0
        self.running_mean = 0.0
        self.running_var = 0.0

    def act(self, obs):

        # masking: Calculate valid moves
        visited_enriched_tensor = self._apply_time_constraints_v3(
            self.fleetStatus.trucklist,
            obs["visited_targets"]
        )
        
        observation_space_as_features = self._get_enriched_observation_space(obs)  
        
        # masking: inactive trucks   
        inactive_trucks_mask = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.bool).to(self.cfg.device)
        
        
        truck, node = self._select_action(
            observation_space_as_features,  
            obs["current_trucks"], 
            visited_enriched_tensor,
            inactive_trucks_mask)
        


        return int(truck), int(node)
        

    def store_reward(self, reward):
        if self.cfg.debug: print(f"DEBUG: Storing reward: {reward}")
        self.rewards.append(reward)

    def update(self):
        """
        Policy Gradient (REINFORCE)
        """        

        n_probs = len(self.log_probs)
        n_rewards = len(self.rewards)
        
        if self.cfg.debug: print(f"DEBUG: Log_Probs: {n_probs} | Rewards: {n_rewards}")

        assert n_probs == n_rewards, \
            f"MISALIGNMENT DETECTED! You have {n_probs} actions but {n_rewards} rewards."

        R = 0
        policy_loss = []
        returns = []

        
   
        # Calculate Returns (Cumulative Reward from t to T)
        # example:
        # Step 	reward	return
        # 3	    -0.2	-0.2
        # 2	    -2.0	-2.2
        # 1	    -0.5	-2.7
        # 0	    -1.0	-3.7
        for r in reversed(self.rewards):
            R = r + self.cfg.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.cfg.device)

        # 1. Calculate Batch Stats
        batch_mean = returns.mean().item()
        
        # If the episode only had 1 step, variance is undefined (NaN). Default to 0.
        if len(returns) > 1:
            batch_var = returns.var(unbiased=False).item()
        else:
            batch_var = 0.0

        # 2. Increment Step for Bias Correction
        self.step_count += 1
        correction_factor = 1.0 - (self.cfg.beta ** self.step_count)

        # 3. Update Moving Averages (EMA)
        self.running_mean = self.cfg.beta * self.running_mean + (1 - self.cfg.beta) * batch_mean
        self.running_var = self.cfg.beta * self.running_var + (1 - self.cfg.beta) * batch_var

        # 4. Apply Bias Correction
        corrected_mean = self.running_mean / correction_factor
        corrected_var = self.running_var / correction_factor

        # 5. Extract Final Standard Deviation
        corrected_sd = np.sqrt(corrected_var)

        # 6. Normalize the Returns!
        normalized_returns = (returns - corrected_mean) / (corrected_sd + 1e-8)
        mean_normalized_return = normalized_returns.mean().item()
        
        for log_prob, R, entropy in zip(self.log_probs, normalized_returns, self.entropies):
            policy_loss.append(-log_prob * R)
            
            
        
        mean_entropy = torch.stack(self.entropies).mean()

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).mean() #each policy_loss item is a scalar tensor, needs stack to sum
        loss = policy_loss - self.cfg.entropy_bonus * mean_entropy  # Add entropy bonus to loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5) # Gradient clipping to prevent exploding gradients
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
       

        return loss.item(), mean_entropy.item(), grad_norm, mean_normalized_return
      
    
    def _get_enriched_observation_space(self, obs):
        """
        Concatenate all observation space elements with dimension N into a single tensor.
        """
        is_target = torch.tensor(obs["is_target"], dtype=torch.float32).unsqueeze(1).to(self.cfg.device)  # Ensure Shape: (N, 1)
        visited_targets = torch.tensor(obs["visited_targets"], dtype=torch.float32).unsqueeze(1).to(self.cfg.device)  # Ensure Shape: (N, 1)
        time_matrix = torch.tensor(obs["time_matrix"], dtype=torch.float32).to(self.cfg.device)  # Shape: (N, N)

        # Concatenate all tensors with dimension N along the last axis
        enriched_tensor = torch.cat([time_matrix,is_target, visited_targets], dim=1)  # Shape: (N, N+2) if time_matrix is (N, N) and the others are (N, 1)

        return enriched_tensor
    
 
    
    
    def _apply_time_constraints_v3(self, truck_list, visited_mask):
        """
        Optimized version of _apply_time_constraints with corrected return time calculation.
        """
        time_matrix = torch.as_tensor(self.fleetStatus.time_matrix, device=self.cfg.device)  # Avoid unnecessary tensor creation
        num_nodes = time_matrix.shape[0]
        visited_mask = torch.tensor(visited_mask, dtype=torch.bool, device=self.cfg.device)  # Ensure visited_mask is a tensor
        masks = visited_mask.clone()  # Start with the visited mask

        # Ensure masks has the correct shape for multiple trucks
        if masks.dim() == 1:
            masks = masks.unsqueeze(0).repeat(len(truck_list), 1)

        # Precompute return times for all nodes to the depot for each truck
        return_times = {
            truck_id: time_matrix[:, truck_state.tour[0]] if truck_state.tour else torch.zeros(num_nodes, device=self.cfg.device)
            for truck_id, truck_state in enumerate(truck_list.values())
        }

        # Iterate over trucks
        for truck_id, truck_state in enumerate(truck_list.values()):
            current_node = truck_state.tour[-1] if truck_state.tour else 0
            current_time = truck_state.total_time

            # Calculate total time for all nodes in a vectorized manner
            travel_times = time_matrix[current_node]
            total_times = current_time + travel_times + return_times[truck_id]

            # Mask nodes that exceed the time constraint
            masks[truck_id] |= total_times > self.cfg.max_daily_delivery_time_each_truck

        return masks        
        
     
                
        
    def _select_action(self, nodes, truck_positions, visited_enriched, inactive_trucks_mask):
        """
        Helper to select the next action or return NO-OP if all nodes are visited.
        """

        # Pass the mask to the policy
        truck_probs, node_probs = self.policy(nodes, truck_positions, visited_enriched, inactive_trucks_mask)
        
        # ---- sample truck ----
        truck_dist = torch.distributions.Categorical(truck_probs)
        truck = truck_dist.sample()

        # ---- sample node for that truck ----
        num_nodes = nodes.shape[0]
        
        if torch.allclose(node_probs[truck], torch.full_like(node_probs[truck], node_probs[truck][0].item())):  # Check if all node probabilities for the selected truck are masked (== -1e9)
            node_probs = torch.zeros(num_nodes + 1, device=nodes.device)  # Create a new tensor for node probabilities
            node_probs[-1] = 1.0  # Assign probability 1 to the NO-OP action at the last index
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