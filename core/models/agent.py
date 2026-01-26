import torch
import torch.optim as optim

from core.models.policy import AttentionPolicy, GraphPointerPolicy


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





class MDVRPREINFORCEAgent:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        # node_dim is the size of the proximity profile (num_nodes)
        self.policy = GraphPointerPolicy(node_dim=data["num_nodes"], embed_dim=cfg.embed_dim)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        node_features = state["node_features"].to(self.cfg.device)
        num_nodes = self.data["num_nodes"]
        
        visited_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.cfg.device)
        for d in self.data["depots"]:
            visited_mask[d.idx] = True # Prevent visiting depots as customers
            
        full_plan = {}
        total_log_prob = 0
        
        # Build routes sequentially for each truck
        for truck in self.data["trucks"]:
            truck_route = []
            current_node = truck.depot_idx # each truck starts at its own depot
            
            # Allow each truck to visit a balanced share of customers
            max_p_truck = (len(self.data["customers"]) // len(self.data["trucks"])) + 50
            
            for _ in range(max_p_truck):
                if visited_mask.all(): break
                    
                probs = self.policy(node_features, current_node, visited_mask)
                if probs.sum() == 0: break
                    
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                total_log_prob += dist.log_prob(action)
                next_node = action.item()
                truck_route.append(next_node)
                visited_mask[next_node] = True
                current_node = next_node
                
            full_plan[truck.id] = truck_route

        self.log_probs.append(total_log_prob)
        return full_plan

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = torch.tensor(self.rewards).to(self.cfg.device)
        # Standardize rewards for stable gradients
        R = (R - R.mean()) / (R.std() + 1e-9)
        
        loss = -(torch.stack(self.log_probs) * R).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs.clear()
        self.rewards.clear()
        return loss.item()
