import torch
import torch.optim as optim
from core.models.policy import AttentionPolicy
from core.models.policy import  GraphPointerPolicy

# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        """
        state = (nodes, current_node, visited_mask)
        """        
        nodes = state["nodes"].to(self.cfg.device)
        visited = state["visited"].to(self.cfg.device)
        current = state["current"]
        
        probs = self.policy(nodes, current, visited)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        """
        Policy Gradient (REINFORCE)
        """        
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
            R = r + R # No discount factor for simple TSP usually, or use 0.99
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.cfg.device)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        return loss.item()


# ----------------------------
# REINFORCEFleetAgent 
# ---------------------------- 
class REINFORCEFleetAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self.log_probs = []
        self.rewards = []

    def act(self, state):
        nodes = state["nodes"].to(self.cfg.device)
        visited_targets = state["visited_targets"].to(self.cfg.device)
        is_target = state["is_target"].to(self.cfg.device)

        actions = []
        step_log_probs = []

        mask = visited_targets | ~is_target

        for k in range(self.cfg.num_trucks):
            current = state["current"][k]

            probs = self.policy(nodes, current, mask)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            actions.append(action.item())
            step_log_probs.append(dist.log_prob(action))

        self.log_probs.append(torch.stack(step_log_probs).sum())
        return actions

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = 0
        returns = []

        for r in reversed(self.rewards):
            R = r + R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.cfg.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = sum(-lp * R for lp, R in zip(self.log_probs, returns))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        return loss.item()