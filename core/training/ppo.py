import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    def __init__(
        self,
        policy,
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
        self.policy = policy.to(device)
        self.old_policy = type(policy)(
            input_dim,
            hidden_dim,
            num_trucks,
            num_nodes
        ).to(device)

        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

  
    def _compute_returns(self, rewards, dones):
        returns = []
        G = 0.0

        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)

        return torch.stack(returns)
    def update(self, batch):
        states = batch["states"].to(self.device)          # (T, N, F)
        truck_actions = batch["truck_actions"].to(self.device)
        node_actions = batch["node_actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        log_probs_old = batch["log_probs"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)

        T, N, F = states.shape

        returns = self._compute_returns(rewards, dones).detach()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        truck_logits_list = []
        node_logits_list = []
        values_list = []

        for t in range(T):
            flat_state = states[t]  # (N, F)
            truck_logits, node_logits, value = self.policy(flat_state, edge_index)

            truck_logits_list.append(truck_logits)
            node_logits_list.append(node_logits)
            values_list.append(value.squeeze())

        truck_logits = torch.stack(truck_logits_list)  # (T, num_trucks)
        node_logits = torch.stack(node_logits_list)    # (T, num_nodes)
        values = torch.stack(values_list)              # (T,)

      
        truck_dist = Categorical(logits=truck_logits)
        node_dist = Categorical(logits=node_logits)

        log_probs = truck_dist.log_prob(truck_actions) + node_dist.log_prob(node_actions)
        entropy = truck_dist.entropy().mean() + node_dist.entropy().mean()

        advantages = returns - values.detach()
        ratio = torch.exp(log_probs - log_probs_old)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        self.old_policy.load_state_dict(self.policy.state_dict())

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }