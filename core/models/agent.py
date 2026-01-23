import torch
import torch.optim as optim
from core.envs.tsp_env import TSPEnv
from core.models.policy import AttentionPolicy

class REINFORCEAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = AttentionPolicy(embed_dim=cfg.embed_dim)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.log_probs = []
        self.rewards = []

    def act(self, state,eval_mode=False):
        nodes = state["nodes"].to(self.cfg.device)
        visited = state["visited"].to(self.cfg.device)
        current = state["current"]
        
        probs = self.policy(nodes, current, visited)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        if eval_mode:
            action = torch.argmax(probs).unsqueeze(0)
        else: 
            action = dist.sample()
            self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate Returns (Cumulative Reward from t to T)
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
        
        self.log_probs = []
        self.rewards = []
        return loss.item()
    def evaluate_mode(self, cfg):
        env = TSPEnv(cfg)
        state, _ = env.reset()
        terminated = False
        total_reward = 0
        route_sequence = [state["current"]]

        self.policy.eval()
        with torch.no_grad():
            while not terminated:
                action = self.act(state, eval_mode=True)
                state, reward, terminated, _, _ = env.step(action)
                total_reward += reward
                route_sequence.append(action)

        return total_reward, route_sequence