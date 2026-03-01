import torch
import torch.optim as optim
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from loader_lib.data_loader import FleetStatus
from .policy import GraphPointerPolicy

# ----------------------------
# PPOAgent 
# ---------------------------- 
class PPOAgent:
        
    def __init__(self, cfg, fleetStatus: FleetStatus):
        self.cfg = cfg
        self.fleetStatus = fleetStatus
        
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim, node_dim=4, cfg=cfg).to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self.returns_var = None  # For storing returns' variance to later normalize them with EMA
        
        # PPO Hyperparameters (Fallbacks added in case they aren't in your cfg)
        self.gamma = getattr(cfg, 'gamma', 0.99)
        self.eps_clip = getattr(cfg, 'eps_clip', 0.2)
        self.ppo_epochs = getattr(cfg, 'ppo_epochs', 4)
        self.entropy_bonus = getattr(cfg, 'entropy_bonus', 0.01)
        self.value_coef = getattr(cfg, 'value_coef', 0.5)

        # PPO Memory Buffers
        self.saved_nodes = []
        self.saved_curr_nodes = []
        self.saved_visited = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.terminal_flags = []

    def act(self, obs):
        # if getattr(self.cfg, 'debug', False): print(f"DEBUG: Observations received: {obs.keys()}")
        # if getattr(self.cfg, 'debug', False): print(f"DEBUG: Nodes: {obs['nodes']}")
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)
        nodes = (nodes - nodes.mean(dim=0)) / (nodes.std(dim=0) + 1e-9)
        # if getattr(self.cfg, 'debug', False): print(f"DEBUG: Nodes mean: {nodes.mean(dim=0)}")
        # raise(Exception("Debugging: Check node normalization"))

        # Calculate valid moves
        visited_enriched = self._apply_time_constraints(
            self.fleetStatus.active_truck,
            self.fleetStatus.trucklist,
            obs["visited_targets"]
        )
        visited_enriched = torch.tensor(visited_enriched, dtype=torch.bool).to(self.cfg.device)
        current_node = obs["current_trucks"][self.fleetStatus.active_truck]

        enhanced_features = self._get_enriched_nodes(nodes, obs["visited_targets"])
        
        # Save states for PPO multiple epochs
        self.saved_nodes.append(enhanced_features.detach())
        self.saved_curr_nodes.append(current_node)
        self.saved_visited.append(visited_enriched.detach())

        action_result = self._select_action(enhanced_features, current_node, visited_enriched)
        if getattr(self.cfg, 'debug', False): print(f"DEBUG: Action result (index): {action_result}")
        return int(action_result)
        
    def store_reward(self, reward, is_terminal):
        if getattr(self.cfg, 'debug', False): print(f"DEBUG: Storing reward: {reward}")
        self.rewards.append(reward)
        self.terminal_flags.append(is_terminal)

    def update(self):
        """
        Proximal Policy Optimization (PPO) Update
        """        
        # Calculate Returns (Cumulative Reward from t to T)
        R = 0
        returns = []
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.terminal_flags)):
            if is_terminal:
                R = 0  # reset return at episode boundaries
            R = reward + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        current_var = returns.var().item()
        if self.returns_var is None:
            self.returns_var = current_var
        else:
            self.returns_var = (1-self.cfg.returns_var_alpha) * self.returns_var + self.cfg.returns_var_alpha * current_var  # EMA update
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # returns = returns * self.cfg.reward_scale  # Scale returns if needed
        returns = returns / (self.returns_var ** 0.5 + 1e-9) # Scale returns using the EMA of variance
        
        # Prepare old data tensors
        old_logprobs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach()
        old_actions = torch.tensor(self.actions, dtype=torch.long).to(self.cfg.device)
        
        # Calculate Advantages
        advantages = returns - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        total_loss, total_entropy, total_grad_norm = 0, 0, 0
        
        # PPO Multiple Epochs Loop
        for _ in range(self.ppo_epochs):
            new_logprobs = []
            new_values = []
            entropies = []
            
            # Re-evaluate the saved states using the UPDATED policy
            for nodes, curr_node, visited in zip(self.saved_nodes, self.saved_curr_nodes, self.saved_visited):
                probs, val = self.policy(nodes, curr_node, visited)
                dist = torch.distributions.Categorical(probs)
                
                new_values.append(val)
                entropies.append(dist.entropy())
                
            # Compute new log probabilities for the EXACT SAME actions taken previously
            new_values = torch.stack(new_values).squeeze()
            entropies = torch.stack(entropies)
            
            # Reconstruct dists to get new log_probs
            for i, (nodes, curr_node, visited) in enumerate(zip(self.saved_nodes, self.saved_curr_nodes, self.saved_visited)):
                 probs, _ = self.policy(nodes, curr_node, visited)
                 dist = torch.distributions.Categorical(probs)
                 new_logprobs.append(dist.log_prob(old_actions[i]))
                 
            new_logprobs = torch.stack(new_logprobs)
            
            # -----------------------------------------
            # Surrogate Loss Calculation
            # -----------------------------------------
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values, returns)
            
            loss = actor_loss + (self.value_coef * critic_loss) - (self.entropy_bonus * entropies.mean())
            
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_entropy += entropies.mean().item()
            total_grad_norm += grad_norm.item()
            
        # Clear Memory Buffers
        self.saved_nodes.clear()
        self.saved_curr_nodes.clear()
        self.saved_visited.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.terminal_flags.clear()

        # Averages over epochs to report back to main.py seamlessly
        return total_loss / self.ppo_epochs, total_entropy / self.ppo_epochs, total_grad_norm / self.ppo_epochs
    
    def _get_enriched_nodes(self, nodes, visited):
        num_nodes = nodes.shape[0]
        # print("trucklist", self.fleetStatus.trucklist)
        # print("trucklist values", list(self.fleetStatus.trucklist.values()))
        # raise(Exception("Debugging: Check trucklist values"))

        truck_positions = [state.position for state in self.fleetStatus.trucklist.values()]
        truck_finishers = [state.finished for state in self.fleetStatus.trucklist.values()]
        active_truck_positions = [truck_position for truck_position, finished in zip(truck_positions, truck_finishers) if not finished]
        is_truck_position = torch.zeros(num_nodes, dtype=torch.float32).to(self.cfg.device)
        # is_noop_position = torch.zeros(num_nodes, dtype=torch.float32).to(self.cfg.device)
        for i, pos in enumerate(active_truck_positions):
            is_truck_position[pos] += 1.0

            # if is_noop_position[i]:
            #     is_noop_position[pos] = 1.0  # Ensure active trucks are marked as such
        
        is_truck_position = is_truck_position / self.fleetStatus.num_trucks()  # Normalize by number of trucks to keep values in a reasonable range
        visited = torch.tensor(visited, dtype=torch.float32).to(self.cfg.device)
        # print(f"DEBUG: visited_enriched: {visited}")
        # print(f"DEBUG: is_truck_position: {is_truck_position}")
        # breakpoint()
        # visited = visited.float().unsqueeze(1)  # Convert boolean mask to float for concatenation
        enriched_nodes = torch.cat([nodes, is_truck_position.unsqueeze(1), visited.unsqueeze(1)], dim=1)
        return enriched_nodes    
    
    def _apply_time_constraints(self, active_truck, trucks_dict_state, visited_mask):
        mask = visited_mask.copy()
        truck_state = trucks_dict_state[active_truck]
        current_node = truck_state.tour[-1] if truck_state.tour else 0

        time_matrix = self.fleetStatus.time_matrix
        num_nodes = time_matrix.shape[0]
        for next_node in range(num_nodes):
            if mask[next_node]:
                continue 
            next_travel_time = time_matrix[current_node, next_node]
            time_to_return = time_matrix[next_node, truck_state.tour[0]] 
            if truck_state.total_time + next_travel_time + time_to_return > self.cfg.max_daily_delivery_time_each_truck:
                # print(f"DEBUG: Masking node {next_node} for truck {active_truck} due to time constraint. Current time: {truck_state.total_time}, Travel time: {next_travel_time}, Time to return: {time_to_return}")
                mask[next_node] = True
        return mask    
        
    def _select_action(self, nodes, current_node, visited_enriched):
        # NEW: Now unpacks the value prediction too
        probs, state_value = self.policy(nodes, current_node, visited_enriched)
        
        if getattr(self.cfg, 'debug', False): print(f"DEBUG: Action probabilities before masking: {(probs.cpu().detach().numpy() * 1000).astype(int)}")
        # if getattr(self.cfg, 'debug', False): print(f"DEBUG: Action probabilities before masking:")
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # print(f"DEBUG: Sampled action index: {action.item()} with probability {probs[action].item():.4f}")
        
         # Store log probability and value for PPO update
        
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value) # Save the critic's assessment

        return action.item()