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
 
        
        self.policy = FactorizedFleetPolicy( embed_dim=cfg.embed_dim, cfg=cfg)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []
        
        self.entropies = []



    def act(self, obs):

        # masking: Calculate valid moves
        visited_enriched_tensor = self._apply_time_constraints_v3(obs)
        
        # enrich observation space concatenating 
        observation_space_as_features = self._get_enriched_observation_space(obs)  
        
        # masking: inactive trucks   
        inactive_trucks_mask = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.bool).to(self.cfg.device)
        
        
        truck, node = self._select_action(
            observation_space_as_features,  
            obs["truck_positions"], 
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
        entropy_loss = []
        
   
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
        
        # Normalize returns for stability
        returns = returns.float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)        


       
        
        for log_prob, R, entropy in zip(self.log_probs, returns, self.entropies):
            policy_loss.append(-log_prob * R)
            entropy_loss.append(entropy)
            
        
      

        self.optimizer.zero_grad()
        
        loss = torch.stack(policy_loss).sum() - self.cfg.entropy_bonus * torch.stack(entropy_loss).sum() #each policy_loss item is a scalar tensor, needs stack to sum
        
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
       

        return loss.item(), 0, 0, 0
      
    
    
    def _get_enriched_observation_space(self, obs):
        """
        Concatenate spatial, status, and fleet context into a fixed-size feature vector.
        Removed the N*N time_matrix to improve generalization.
        """
        device = self.cfg.device
        
        # 1. Coordinates (Spatial information) - Shape: [N, 2]
        coords = torch.tensor(obs["nodes"], dtype=torch.float32, device=device)
        num_nodes = coords.shape[0]

        # 2. Node Status - Shape: [N, 1] each
        is_target = torch.tensor(obs["is_target"], dtype=torch.float32, device=device).unsqueeze(1)
        visited = torch.tensor(obs["visited_targets"], dtype=torch.float32, device=device).unsqueeze(1)

        # 3. Global Fleet Context (Broadcasted) - Shape: [N, 3]
        inactive_mask = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.float32, device=device)
        truck_times = torch.tensor(obs["truck_times"], dtype=torch.float32, device=device)
        
        active_ratio = (1.0 - inactive_mask).mean().reshape(1, 1)
        avg_fleet_time = truck_times.mean().reshape(1, 1)
        max_fleet_time = truck_times.max().reshape(1, 1)
        
        fleet_stats = torch.cat([active_ratio, avg_fleet_time, max_fleet_time], dim=1).repeat(num_nodes, 1)

        # 4. Depot / Home Information
        truck_starts = obs["truck_starts"] # List of depot indices
        
        # Feature: Is this node a home depot? (And how many trucks live there)
        home_counts = torch.zeros(num_nodes, 1, device=device)
        for start_idx in truck_starts:
            home_counts[start_idx] += 1
            
        # Feature: Proximity to safety (Distance to nearest depot)
        time_matrix = torch.tensor(obs["time_matrix"], dtype=torch.float32, device=device)
        depot_indices = list(set(truck_starts))
        dist_to_depots = time_matrix[:, depot_indices]
        min_dist_to_depot, _ = torch.min(dist_to_depots, dim=1, keepdim=True)

        # 5. Final Concatenation - Total Dimension: 9
        # [Coords(2), Target(1), Visited(1), Fleet(3), Home(1), MinDist(1)]
        enriched_tensor = torch.cat([
            coords,
            is_target,
            visited,
            fleet_stats,
            home_counts,
            min_dist_to_depot
        ], dim=1)

        return enriched_tensor


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
        Optimized version of _apply_time_constraints with corrected return time calculation.
        """
        visited_mask = obs["visited_targets"]
        time_matrix = obs["time_matrix"]
        truck_positions = obs["truck_positions"]
        truck_starts = obs["truck_starts"]
        truck_times = obs["truck_times"]        
        
        time_matrix = torch.as_tensor(time_matrix, device=self.cfg.device)  # Avoid unnecessary tensor creation
        visited_mask = torch.tensor(visited_mask, dtype=torch.bool, device=self.cfg.device)  # Ensure visited_mask is a tensor
        masks = visited_mask.clone()  # Start with the visited mask

        # Ensure masks has the correct shape for multiple trucks
        if masks.dim() == 1:
            masks = masks.unsqueeze(0).repeat(len(truck_positions), 1)

        # Precompute return times from all nodes to the depot for each truck
        return_times_from_next_nodes = {
            truck_id: time_matrix[:, truck_starts[truck_id]]
            for truck_id in range(len(truck_starts))
        }

        # Iterate over trucks using truck_positions
        for truck_id, current_node in enumerate(truck_positions):
            current_time = truck_times[truck_id]

            # Calculate travel times from the current node to all other nodes
            travel_times_to_next_nodes = time_matrix[current_node]

            # Use precomputed return times
            total_times = current_time + travel_times_to_next_nodes + return_times_from_next_nodes[truck_id]

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