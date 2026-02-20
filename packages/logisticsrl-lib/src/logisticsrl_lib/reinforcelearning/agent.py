import torch
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
from loader_lib.data_loader import FleetStatus
from .policy import GraphPointerPolicy

# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:


        
    def __init__(self, cfg, fleetStatus: FleetStatus):
        self.cfg = cfg
        self.fleetStatus = fleetStatus
        
        
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim, cfg=cfg)
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
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)

        # masking: Calculate valid moves
        visited_enriched = self._apply_time_constraints(
            self.fleetStatus.active_truck,
            self.fleetStatus.trucklist,
            obs["visited_targets"]
        )
        visited_enriched = torch.tensor(visited_enriched, dtype=torch.bool).to(self.cfg.device)
        current_node = obs["current_trucks"][self.fleetStatus.active_truck]

        enhanced_features = self._get_enriched_nodes(nodes)
        
        action_result = self._select_action(enhanced_features, current_node, visited_enriched)
        return int(action_result)
        

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
        
        for log_prob, R in zip(self.log_probs, normalized_returns):
            policy_loss.append(-log_prob * R)
        
        mean_entropy = torch.stack(self.entropies).mean()

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() #each policy_loss item is a scalar tensor, needs stack to sum
        loss = policy_loss - self.cfg.entropy_bonus * mean_entropy  # Add entropy bonus to loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5) # Gradient clipping to prevent exploding gradients
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()

        return loss.item(), mean_entropy.item(), grad_norm
    
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
    
    def _apply_time_constraints(self, active_truck, trucks_dict_state, visited_mask):
            """
            Modify the visited_mask to also mask nodes that would cause the truck to exceed 24h total time if visited.
            """
            mask = visited_mask.copy()  # Start with the original visited mask (targets already visited)
            # get truck state
            truck_state = trucks_dict_state[active_truck]
            current_node = truck_state.tour[-1] if truck_state.tour else 0

            time_matrix = self.fleetStatus.time_matrix
            num_nodes = time_matrix.shape[0]
            for next_node in range(num_nodes):
                if mask[next_node]:
                    continue  # already masked as visited
                next_travel_time = time_matrix[current_node, next_node]
                time_to_return = time_matrix[next_node, truck_state.tour[0]]   # time to return to depot from next node
                if truck_state.total_time + next_travel_time + time_to_return > self.cfg.max_daily_delivery_time_each_truck:
                    mask[next_node] = True
            return mask    
        
    def _select_action(self, nodes, current_node, visited_enriched):
        """
        Helper to select the next action or return NO-OP if all nodes are visited.
        """
        probs = self.policy(nodes, current_node, visited_enriched)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.entropies.append(dist.entropy())
        self.log_probs.append(dist.log_prob(action))

        return action.item()        