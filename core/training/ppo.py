import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, policy_class, input_dim, hidden_dim, num_trucks, num_nodes,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, 
                 value_coef=0.5, entropy_coef=0.1, device="cpu", ppo_epochs=10):
        
        self.device = device
        self.policy = policy_class(input_dim, hidden_dim, num_trucks, num_nodes).to(device)
        
        # Split learning rates: Let the critic learn faster but more carefully
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs

    def update(self, batch):
        states = batch["states"].to(self.device)
        node_actions = batch["node_actions"].to(self.device)
        log_probs_old = batch["log_probs_old"].to(self.device)
        # CRITICAL: Scale rewards down for the critic to handle
        rewards = batch["rewards"].to(self.device)
        masks = batch["masks"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)
        node_masks = batch["node_masks"].to(self.device)

        with torch.no_grad():
            _, _, values = self.policy(states, edge_index)
            values = values.squeeze(-1)
            
            # --- GAE Calculation (Reduces Variance) ---
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0
            for t in reversed(range(len(rewards))):
                next_val = values[t+1] if t + 1 < len(rewards) and masks[t] > 0 else 0
                delta = rewards[t] + self.gamma * next_val * masks[t] - values[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * masks[t] * last_gae_lam
            
            returns = advantages + values
            # Advantage Normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Record old values for Value Clipping
        old_values = values.clone()

        for _ in range(self.ppo_epochs):
            _, node_logits, values_new = self.policy(states, edge_index)
            values_new = values_new.squeeze(-1)

            node_logits = node_logits.masked_fill(node_masks, -1e10)
            dist = Categorical(logits=node_logits)
            
            log_probs = dist.log_prob(node_actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # Policy Loss
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Value Clipping (Prevents Loss Explosion) ---
            v_loss_unclipped = (values_new - returns) ** 2
            v_clipped = old_values + torch.clamp(values_new - old_values, -self.eps_clip, self.eps_clip)
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            # Strict Gradient Clipping
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) 
            self.optimizer.step()

        return {"loss": loss.item(), "value_loss": value_loss.item(), "policy_loss": policy_loss.item()}