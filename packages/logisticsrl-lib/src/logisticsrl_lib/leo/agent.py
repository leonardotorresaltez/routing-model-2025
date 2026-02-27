import torch
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
from loader_lib.data_loader import FleetStatus
from .policy import FactorizedFleetPolicy

# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:


        
    def __init__(self, cfg, fleetStatus: FleetStatus):
        self.cfg = cfg
        self.fleetStatus = fleetStatus
        
        
        self.policy = FactorizedFleetPolicy(node_dim= (fleetStatus.num_nodes() + 2), embed_dim=cfg.embed_dim, cfg=cfg)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []
        
        self.entropies = []
        self.entropy_coef = 0.01

    def act(self, obs):
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)

        # masking: Calculate valid moves
        visited_enriched_tensor = self._apply_time_constraints_v3(
            self.fleetStatus.trucklist,
            obs["visited_targets"]
        )
        

        enhanced_features = self._get_enriched_observation_space(obs)
        
        truck_positions = obs["current_trucks"]
        inactive_trucks_mask = torch.tensor(obs["inactive_trucks_mask"], dtype=torch.bool).to(self.cfg.device)
        
        
        truck, node = self._select_action(enhanced_features, truck_positions, visited_enriched_tensor,inactive_trucks_mask)
        


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
        gamma = 0.95
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
            R = r + gamma * R # No discount factor for simple TSP usually, or use 0.99
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.cfg.device)
        # Normalize returns for stability
        returns = returns.float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R, entropy in zip(self.log_probs, returns, self.entropies):
            policy_loss.append(-log_prob * R)
            entropy_loss.append(entropy)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() - self.entropy_coef * torch.stack(entropy_loss).sum() #each policy_loss item is a scalar tensor, needs stack to sum
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        losint = loss.item()

        return losint
    
    def _get_enriched_nodes(self, nodes):
        """
        Devuelve el tensor de nodos enriquecido con una columna extra que indica si el nodo es la posición actual de algún camión.
        """  
        num_nodes = nodes.shape[0]
        truck_positions = [state.position for state in self.fleetStatus.trucklist.values()]
        is_truck_position = torch.zeros(num_nodes, dtype=torch.float32).to(self.cfg.device)
        for pos in truck_positions:
            if 0 <= pos < num_nodes:
                is_truck_position[pos] = 1.0

        enriched_nodes = torch.cat([nodes, is_truck_position.unsqueeze(1)], dim=1)
        return enriched_nodes    
    
    def _get_enriched_observation_space(self, obs):
        """
        Concatenate all observation space elements with dimension N into a single tensor.
        """
        # nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)  # Shape: (N, 2)
        is_target = torch.tensor(obs["is_target"], dtype=torch.float32).unsqueeze(1).to(self.cfg.device)  # Ensure Shape: (N, 1)
        visited_targets = torch.tensor(obs["visited_targets"], dtype=torch.float32).unsqueeze(1).to(self.cfg.device)  # Ensure Shape: (N, 1)
        time_matrix = torch.tensor(obs["time_matrix"], dtype=torch.float32).to(self.cfg.device)  # Shape: (N, N)

        # Concatenate all tensors with dimension N along the last axis
        enriched_tensor = torch.cat([time_matrix,is_target, visited_targets], dim=1)  # Shape: (N, 502) if time_matrix is (N, N) and the others are (N, 1)

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
            #print("Todos los valores son iguales ")
            node_probs = torch.zeros(num_nodes + 1, device=nodes.device)  # Create a new tensor for node probabilities
            node_probs[-1] = 1.0  # Assign probability 1 to the NO-OP action at the last index

        else:
            #print("No todos los valores son iguales a -1e9")
            node_probs = node_probs[truck]
        
        node_dist = torch.distributions.Categorical(node_probs)
        node = node_dist.sample()

        # ---- log prob joint ----
        log_prob = truck_dist.log_prob(truck) + node_dist.log_prob(node)
        self.log_probs.append(log_prob)
        entropy = truck_dist.entropy() + node_dist.entropy()
        self.entropies.append(entropy)
        return truck.item(), node.item()