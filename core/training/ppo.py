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
        lr=5e-5,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.1,
        device="cpu",
        ppo_epochs=10
    ):
        self.device = device
        self.policy = policy_class(input_dim, hidden_dim, num_trucks, num_nodes).to(device)
        self.old_policy = policy_class(input_dim, hidden_dim, num_trucks, num_nodes).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs

    def select_action(self, state, edge_index, node_masks):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            _, node_logits, value = self.policy(state, edge_index)
            if node_masks is not None:
                node_logits[node_masks] = -1e10

            node_dist = Categorical(logits=node_logits)
            node_actions = node_dist.sample()
            log_prob = node_dist.log_prob(node_actions).sum()

        return node_actions.cpu().numpy(), log_prob.item(), value.item()
    def _compute_returns(self, rewards, masks):
        
        returns = []
        G = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            G = r + self.gamma * G * m
            returns.insert(0, G)
        return torch.FloatTensor(returns).to(self.device)

    def update(self, batch):
        states = batch["states"].to(self.device)
        node_actions = batch["node_actions"].to(self.device)
        log_probs_old = batch["log_probs_old"].to(self.device)
        rewards = batch["rewards"]
        masks = batch["masks"]
        edge_index = batch["edge_index"].to(self.device)

        returns = self._compute_returns(rewards, masks)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            _, node_logits, values = self.policy(states, edge_index)
            values = values.squeeze()
            node_dist = Categorical(logits=node_logits)
            log_probs = node_dist.log_prob(node_actions).sum(dim=-1)
            entropy = node_dist.entropy().mean()
            advantages = (returns - values.detach())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_probs - log_probs_old)
           
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }