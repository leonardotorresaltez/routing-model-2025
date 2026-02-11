import torch
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
from loader_lib.data_loader import FleetStatus
from .policy import GraphPointerPolicy

# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:


        
    def __init__(self, cfg, fleetStatus: FleetStatus):
        self.cfg = cfg
        self.fleetStatus = fleetStatus
        
        
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []

    def act(self, obs):
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)

        # masking: Calculate valid moves
        visited_enriched = self._apply_time_constraints(
            self.fleetStatus.active_truck,
            self.fleetStatus.trucklist,
            obs["visited_targets"]
        )
        visited_enriched = torch.tensor(visited_enriched, dtype=torch.bool).to(self.cfg.device)
        current_node = obs["current_trucks"][self.fleetStatus.active_truck]

        enhanced_features = self._get_enriched_nodes(nodes)
        
        action_result = self._select_action(enhanced_features, current_node, visited_enriched)
        return int(action_result)
        

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
        returns = returns.float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() #each policy_loss item is a scalar tensor, needs stack to sum
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        losint = loss.item()

        return losint
    
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
        
    def _select_action(self, nodes, current_node, visited_enriched):
        """
        Helper to select the next action or return NO-OP if all nodes are visited.
        """
        if visited_enriched.all():
            # NO-OP action, do not call policy as it gets confused
            return self.fleetStatus.num_nodes()  
        probs = self.policy(nodes, current_node, visited_enriched)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()        