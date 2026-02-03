import torch
import torch.optim as optim

import torch.nn.functional as F

from core.models.policy import  GraphPointerPolicy, MultiHeadGraphPointerPolicy

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

    def act(self,obs, active_truck):
     
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)
        visited = torch.tensor(obs["visited_targets"], dtype=torch.bool).to(self.cfg.device)

        
        current_node = obs["current_trucks"][active_truck]
        
        probs = self.policy(nodes, current_node, visited)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())


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