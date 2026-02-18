import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    def __init__(
        self,
        policy_class, 
        input_dim,
        hidden_dim,
        num_trucks,
        num_nodes,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.device = device
        
        # Initialize Actor-Critic
        self.policy = policy_class(input_dim, hidden_dim, num_trucks, num_nodes).to(device)
        self.old_policy = policy_class(input_dim, hidden_dim, num_trucks, num_nodes).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state, edge_index, truck_mask=None, node_mask=None):
        """Selects action using the current policy with optional masking."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            # Policy should return: truck_logits, node_logits, state_value
            truck_logits, node_logits, value = self.policy(state, edge_index)

            # Apply masks if provided (setting logit to a very large negative)
            if truck_mask is not None:
                truck_logits[truck_mask] = -1e10
            if node_mask is not None:
                node_logits[node_mask] = -1e10

            truck_dist = Categorical(logits=truck_logits)
            node_dist = Categorical(logits=node_logits)

            truck_act = truck_dist.sample()
            node_act = node_dist.sample()

            log_prob = truck_dist.log_prob(truck_act) + node_dist.log_prob(node_act)

        return (truck_act.item(), node_act.item()), log_prob.item(), value.item()

    def _compute_returns(self, rewards, masks):
        """Computes discounted rewards (Returns)."""
        returns = []
        G = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            G = r + self.gamma * G * m
            returns.insert(0, G)
        return torch.FloatTensor(returns).to(self.device)

    def update(self, batch):
        # batch should contain: states, truck_acts, node_acts, log_probs_old, rewards, masks, edge_index
        states = batch["states"].to(self.device)
        truck_actions = batch["truck_actions"].to(self.device)
        node_actions = batch["node_actions"].to(self.device)
        log_probs_old = batch["log_probs_old"].to(self.device)
        rewards = batch["rewards"]
        masks = batch["masks"] # (1 - terminated)
        edge_index = batch["edge_index"].to(self.device)

        # 1. Compute Returns & Advantages
        returns = self._compute_returns(rewards, masks)
        
        # 2. Get current policy output
        # Re-running the policy for the whole batch
        truck_logits, node_logits, values = self.policy(states, edge_index)
        values = values.squeeze()

        # 3. Calculate Log Probs and Entropy for Multi-Discrete
        truck_dist = Categorical(logits=truck_logits)
        node_dist = Categorical(logits=node_logits)
        
        log_probs = truck_dist.log_prob(truck_actions) + node_dist.log_prob(node_actions)
        entropy = (truck_dist.entropy() + node_dist.entropy()).mean()

        # 4. PPO Loss
        advantages = (returns - values.detach())
        # Standardize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, returns)

        # 5. Total Loss & Optimization
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }