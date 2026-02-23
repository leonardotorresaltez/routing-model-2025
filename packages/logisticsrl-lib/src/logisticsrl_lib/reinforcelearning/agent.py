import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loader_lib.data_loader import FleetStatus

from .policy import GraphPointerPolicy


# ----------------------------
# PPOAgent 
# ---------------------------- 
class PPOAgent:


        
    def __init__(self, cfg, fleetStatus: FleetStatus):
        self.cfg = cfg
        self.fleetStatus = fleetStatus
        self.values = []
        
        
        # self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim, cfg=cfg)
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim, cfg=cfg, node_dim=4)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []
        self.entropies = []

        self.step_count = 0
        self.running_mean = 0.0
        self.running_var = 0.0
        
        self.memory_nodes = []
        self.memory_trucks = []
        self.memory_masks = []
        self.memory_actions = []
        
        self.episode_boundaries = [0]
        self.episode_count = 0


    def act(self, obs):
       
        # Pass mask to policy for masking
        action_result = self._select_action(
            obs
        )
        return int(action_result)



    def store_reward(self, reward):
        if self.cfg.debug: print(f"DEBUG: Storing reward: {reward}")
        self.rewards.append(reward)

    def finalize_episode(self):
        """Mark the end of an episode and update episode tracking"""
        self.episode_boundaries.append(len(self.rewards))
        self.episode_count += 1

    def should_update_batch(self):
        """Check if we have enough episodes collected for a batch update"""
        return self.episode_count >= self.cfg.batch_episodes

    def update(self):
        """
        Proximal Policy Optimization (PPO) Update
        """        
        if len(self.rewards) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Fallbacks just in case they aren't in config yet
        ppo_epochs = getattr(self.cfg, 'ppo_epochs', 4)
        ppo_clip = getattr(self.cfg, 'ppo_clip', 0.2)
        gae_lambda = getattr(self.cfg, 'gae_lambda', 0.95)

        # 1. Prepare Values
        values_tensor = torch.stack(self.values).squeeze().detach()
        if values_tensor.dim() == 0: 
            values_tensor = values_tensor.unsqueeze(0)

        # 2. Calculate GAE (Generalized Advantage Estimation) PER-EPISODE
        advantages = []
        returns = []
        
        for ep_idx in range(len(self.episode_boundaries) - 1):
            start = self.episode_boundaries[ep_idx]
            end = self.episode_boundaries[ep_idx + 1]
            
            gae = 0
            ep_advantages = []
            ep_returns = []
            
            for i in reversed(range(start, end)):
                next_val = values_tensor[i + 1] if i + 1 < end else 0.0
                delta = self.rewards[i] + self.cfg.gamma * next_val - values_tensor[i]
                gae = delta + self.cfg.gamma * gae_lambda * gae
                
                ep_advantages.insert(0, gae)
                ep_returns.insert(0, gae + values_tensor[i])
            
            advantages.extend(ep_advantages)
            returns.extend(ep_returns)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        # old_log_probs = torch.stack(self.log_probs).to(self.cfg.device)
        old_log_probs = torch.stack(self.log_probs).to(self.cfg.device).detach()  # Add .detach() here!

        # Normalize Returns to stabilize the Critic
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = advantages.detach() # Add this line!
        returns = returns.detach()       # Add this line too!

        # Variables for logging
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        # 3. PPO EPOCHS (Reuse the episode data multiple times)
        for _ in range(ppo_epochs):
            new_log_probs = []
            new_values = []
            new_entropies = []

            # Re-evaluate all saved states sequentially to avoid breaking GNN dimensions
            for n, t, m, a in zip(self.memory_nodes, self.memory_trucks, self.memory_masks, self.memory_actions):
                probs, val = self.policy(n, t, m)
               
                safe_probs = probs.clamp(min=1e-8)
                dist = torch.distributions.Categorical(safe_probs)
            
                new_log_probs.append(dist.log_prob(a))
                new_values.append(val)
                new_entropies.append(dist.entropy())

            new_log_probs = torch.stack(new_log_probs).view(-1)
            # new_values = torch.stack(new_values).squeeze()
            # if new_values.dim() == 0: new_values = new_values.unsqueeze(0)
            
            mean_entropy = torch.stack(new_entropies).mean()

            # 4. PPO Ratio = exp(new_log_prob - old_log_prob)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 5. Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
            
            # Actor Loss (Negative because we want to maximize surrogate)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            new_values = torch.stack(new_values).view(-1)
            returns = returns.view(-1)
            # Critic Loss (MSE between new predictions and actual returns)
            value_loss_unclipped = F.mse_loss(new_values, returns, reduction="none")
            # Clipped MSE Loss (Prevents the Critic from updating too fast)
            value_clipped = values_tensor + torch.clamp(
                new_values - values_tensor, 
                -ppo_clip, 
                ppo_clip
            )
            value_loss_clipped = F.mse_loss(value_clipped, returns, reduction="none")
            # Take the maximum of both (which forces the loss to be bounded by the clip)
            critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Total Loss
            loss = actor_loss + 0.5 *critic_loss - self.cfg.entropy_bonus * mean_entropy

            # 6. Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Accumulate for logging
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += mean_entropy.item()

        # Clear Memory Buffers
        self.memory_nodes.clear()
        self.memory_trucks.clear()
        self.memory_masks.clear()
        self.memory_actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

        # Averages for logging
        avg_loss = (total_actor_loss + 1 * total_critic_loss) / ppo_epochs # The Actor Loss in PPO doesn't drop to zero. It constantly compares the new policy to the old policy to find small, clipped improvements (Advantages). Because the "baseline" (old policy) shifts every update, this loss tends to oscillate around zero rather than steadily decreasing.
        avg_entropy = total_entropy / ppo_epochs # Entropy measures the agent's randomness (exploration). While it will slowly decrease as the agent becomes more confident in its routes, the entropy_bonus parameter intentionally pushes against this to prevent it from dropping too fast and getting stuck in local optima.
        avg_critic = total_critic_loss / ppo_epochs # calculates the Mean Squared Error (MSE) between what the Value Network predicted the route would score versus the actual reward it got. As the network sees more routes, it naturally gets better at predicting the outcome, so this error smoothly drops toward zero. This proves your shared Graph Neural Network is successfully learning the environment.

        self.episode_boundaries = [0]
        self.episode_count = 0
        
        return avg_loss, avg_entropy, avg_critic, returns.mean().item()

    
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
        
    def _select_action(self, obs):
        """
        Select action for multi-truck scenario.
        Action = (truck_id, customer_idx) encoded as single integer.
        """
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.uint8, device=self.cfg.device)        
        current_trucks = torch.tensor(obs["current_trucks"], dtype=torch.long, device=self.cfg.device)
        
        is_target = torch.tensor(obs["is_target"], dtype=torch.float32).to(self.cfg.device).unsqueeze(1)
        visited = torch.tensor(obs["visited_targets"], dtype=torch.float32).to(self.cfg.device).unsqueeze(1)
        
        # Combine into a single tensor of shape [N, 4]: [lat, lon, is_target, visited]
        enhanced_nodes = torch.cat([nodes, is_target, visited], dim=1)
        
        
        # Check if all actions are masked
        if action_mask.sum() == action_mask.shape[0]:
            # print(f"WARNING: All actions masked. Terminating episode.")
            return -1  # Return invalid action code to signal early termination
        
        
        # Pass the enhanced_nodes to the policy instead of the raw coordinates
        probs, value  = self.policy(enhanced_nodes, current_trucks, action_mask)  # Output: [T*N]      
        # # Pass action_mask with correct size: num_trucks * num_nodes
        # probs = self.policy(nodes, current_trucks, action_mask)  # Output: [T*N]
        self.values.append(value)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        
        # Prevent log(0) = -inf by clamping probabilities to a tiny number AFTER sampling
        safe_probs = probs.clamp(min=1e-8)
        safe_dist = torch.distributions.Categorical(safe_probs)
        
        # --- NEW PPO MEMORY SAVING ---
        # Detach them so gradients don't leak between steps
        self.memory_nodes.append(enhanced_nodes.detach())
        self.memory_trucks.append(current_trucks.detach())
        self.memory_masks.append(action_mask.detach())
        self.memory_actions.append(action.detach())
        
        # PPO requires the old log_prob to be completely detached from the graph
        self.log_probs.append(safe_dist.log_prob(action).detach())
        
        return int(action)    