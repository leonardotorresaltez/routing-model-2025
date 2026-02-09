import torch
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
from core.models.policy import  GraphPointerPolicy

# ----------------------------
# REINFORCEAgent 
# ---------------------------- 
class REINFORCEAgent:


        
    def __init__(self, cfg, time_matrix):
        self.cfg = cfg
        self.time_matrix = time_matrix
        self.policy = GraphPointerPolicy(embed_dim=cfg.embed_dim)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        # Buffers for REINFORCE
        self.log_probs = []
        self.rewards = []

    def act(self,obs, active_truck, trucks_dict_state ):
     
        nodes = torch.tensor(obs["nodes"], dtype=torch.float32).to(self.cfg.device)
        # visited = torch.tensor(obs["visited_targets"], dtype=torch.bool).to(self.cfg.device)

        # masking: Calculate valid moves
        visited_targets_copy = obs["visited_targets"]  # Start with the original visited mask (targets already visited)
        # print("visited_enriched antes de contaarrrr: ", visited_targets_copy)
        unvisited_count = (visited_targets_copy == False).sum().item()
        # print(f"Unvisited targets count before time constraintssssss: {unvisited_count}")
        visited_enriched = self._apply_time_constraints(active_truck, trucks_dict_state, visited_targets_copy)
        unvisited_count_2 = (visited_enriched == False).sum().item()
        # print(f"Unvisited targets count after time constraintssssss: {unvisited_count_2}")
        # Comparar cantidad de True
        
        #n_true_mask = sum(visited_targets_copy)
        #n_true_enriched = sum(visited_enriched)
        #if n_true_enriched < n_true_mask:
        #    print(f"visited_mask True: {n_true_mask}, visited_enriched True: {n_true_enriched}")

        visited_enriched = torch.tensor(visited_enriched, dtype=torch.bool).to(self.cfg.device)
        
        current_node = obs["current_trucks"][active_truck]
        #np.set_printoptions(threshold=np.inf)

       
        action_result = -1
        if visited_enriched.all():
            #print("visited_enriched está completamente llenooooooooo de True")
            action_result = nodes.shape[0]  # Acción de NO-OP, apuntando a un índice fuera del rango de nodos, no lamar al policy xq se vuelve loco
        else:
            probs = self.policy(nodes, current_node, visited_enriched)
            dist = torch.distributions.Categorical(probs)
        
            action = dist.sample()
        
            #print("actionnnnn:", action.item())
            #print("visited_enriched luegoo dee MPL: ", visited_targets_copy[action.item()])

            self.log_probs.append(dist.log_prob(action))
            action_result = action.item()

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
        
        if len(self.log_probs) == 0:
            print("No log probabilities stored!!!!!")
        if len(self.rewards) == 0:
            print("No rewards stored. !!!!!")    
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
        #if abs(losint) < 1e-6:
            #print("loss:", losint)
            #print("log_probs:", self.log_probs)
            #print("returns:", returns)
            #print("policy_loss:", policy_loss)
        return losint
    
    def _apply_time_constraints(self, active_truck, trucks_dict_state, visited_mask):
            """
            Modifica la máscara visited_mask para enmascarar también los nodos a los que, si el camión fuera, superaría 24h de tiempo total.
            """
            mask = visited_mask.copy()  # Start with the original visited mask (targets already visited)
            # Obtener el estado actual del camión
            truck_state = trucks_dict_state[active_truck]
            current_node = truck_state.tour[-1] if truck_state.tour else 0
            # Se asume que tienes acceso a la matriz de tiempos (debes pasarla si no está en self)
            # Aquí se asume que self.cfg.time_matrix existe y es un np.array o torch.Tensor
            time_matrix = self.time_matrix
            num_nodes = time_matrix.shape[0]
            for next_node in range(num_nodes):
                if mask[next_node]:
                    continue  # Ya está enmascarado
                next_travel_time = time_matrix[current_node, next_node]
                time_to_return = time_matrix[next_node, truck_state.tour[0]]   # Tiempo de regreso al depósito
                if truck_state.total_time + next_travel_time + time_to_return > self.cfg.max_daily_delivery_time_each_truck:
                    mask[next_node] = True
            return mask    