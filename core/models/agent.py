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
        node_features = state["node_features"].to(self.cfg.device)
        current_node = state["current_node"]
        current_time = state["current_time"]
        visited_mask = state["visited_mask"].to(self.cfg.device)
        active_truck_id = state["active_truck_id"]
        
        # Get the depot index for the active truck
        truck_obj = next(t for t in self.data["trucks"] if t.id == active_truck_id)
        depot_idx = truck_obj.depot_idx
        
        # Create a stricter mask based on time constraints
        # A customer is only 'choosable' if: Time_Now + Time_to_Customer + Time_Back_to_Depot <= Limit
        constraint_mask = visited_mask.clone()
        for i in range(self.data["num_nodes"]):
            if not constraint_mask[i]:
                time_to_i = self.data["time_matrix"][current_node, i].item()
                time_home = self.data["time_matrix"][i, depot_idx].item()
                if current_time + time_to_i + time_home > self.cfg.max_daily_delivery_time_each_truck:
                    constraint_mask[i] = True
        
        probs = self.policy(node_features, current_node, constraint_mask)
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
    """
    This is the Decision Maker that uses the brain.

    Action: It calls the Policy repeatedly to build complete routes for all 50 trucks.
    Memory: It stores the log_probs (how confident it was in its choices) and the rewards received.
    Update: It uses the REINFORCE algorithm. If a journey had a good reward (short time), it "strengthens" the brain to make those choices more likely in the future.
    """
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
            truck_time = 0.0 # Track current time for this truck
            current_node = truck.depot_idx # each truck starts at its own depot
            
            while True:
                if visited_mask.all(): break # All customers done, stop this truck
                
                # refered to max_daily_delivery_time_each_truck
                current_truck_mask = visited_mask.clone()               

                for i in range(num_nodes):
                    if not current_truck_mask[i]: # If not already visited
                        # Time to reach customer 'i' + time to return to depot from 'i'
                        time_to_i = self.data["time_matrix"][current_node, i].item()
                        time_home = self.data["time_matrix"][i, truck.depot_idx].item()
                        
                        if truck_time + time_to_i + time_home > self.cfg.max_daily_delivery_time_each_truck: # it makes sure that it goes back home
                            current_truck_mask[i] = True # Mask this node for this truck since the solution is over the time allowed
                
                if current_truck_mask.all(): break # This truck is "done" because no customers are reachable/unvisited
               
                probs = self.policy(node_features, current_node, current_truck_mask) # It looks at all nodes and calculates a "preference" score for each node based on the truck's
                dist = torch.distributions.Categorical(probs) # This takes the list of "preferences" and puts them available
                action = dist.sample() # One of the prefered is randomly choosen. If we always picked the #1 choice (the biggest slice), it would never try anything new
                total_log_prob += dist.log_prob(action) # It calculates the logarithm of the probability of the choice just made, and sums the logs of all decisions made across the whole day - ONESHOTAPPROACH
                next_node = action.item()
                truck_time += self.data["time_matrix"][current_node, next_node].item()

                truck_route.append(next_node)
                visited_mask = visited_mask.clone()
                visited_mask[next_node] = True
                current_node = next_node
                
            full_plan[truck.id] = truck_route

        self.log_probs.append(total_log_prob)
        return full_plan

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + R # No discount factor for simple TSP usually, or use 0.99
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.cfg.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        loss = -(torch.stack(self.log_probs) * returns).sum()
        # R = torch.tensor(self.rewards).to(self.cfg.device)
        # # Standardize rewards for stable gradients
        # R = (R - R.mean()) / (R.std() + 1e-9)
        
        # # REINFORCE: we need to update the policy weights by the gradient of the log-probability of the actions, scaled by the reward. 
        # loss = -(torch.stack(self.log_probs) * R).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs.clear()
        self.rewards.clear()
        return loss.item()



class MDVRP_one_agent_per_truck_REINFORCE_agent:
    """
    This is the Decision Maker that uses the brain.

    Action: It calls the Policy repeatedly to build complete routes for all 50 trucks.
    Memory: It stores the log_probs (how confident it was in its choices) and the rewards received.
    Update: It uses the REINFORCE algorithm. If a journey had a good reward (short time), it "strengthens" the brain to make those choices more likely in the future.
    """
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        
        # Use data["trucks"] instead of self.trucks
        self.truck_id_to_idx = {t.id: i for i, t in enumerate(data["trucks"])}
        
        # num_nodes + 1 (depot_dist) + num_trucks (fleet_status) + num_trucks (one-hot identity)
        input_dim = data["num_nodes"] + 1 + (len(data["trucks"]) * 2)
        
        self.policy = GraphPointerPolicy(node_dim=input_dim, embed_dim=cfg.embed_dim)
        self.policy.to(cfg.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        
        self.log_probs = []
        self.rewards = []




    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = torch.tensor(self.rewards).to(self.cfg.device)
        # Standardize rewards for stable gradients
        R = (R - R.mean()) / (R.std() + 1e-9)
        
        # REINFORCE: we need to update the policy weights by the gradient of the log-probability of the actions, scaled by the reward. 
        loss = -(torch.stack(self.log_probs) * R).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs.clear()
        self.rewards.clear()
        return loss.item()
    
    def act(self, state):
        if state.get("active_truck_id") is None: return None

        # Prepare Features
        node_features = state["node_features"].to(self.cfg.device)
        depot_dist = state["depot_dist"].to(self.cfg.device).view(-1, 1)
        
        # Fleet context (Broadcast to all nodes)
        fleet_info = torch.cat([state["fleet_status"], state["truck_identity"]]).to(self.cfg.device)
        global_context = fleet_info.repeat(node_features.size(0), 1)
        
        
        # the agent needs these extra features to understand:
        # -Which truck it is (since different trucks have different starting depots).
        # -How much time it has left compared to the rest of the fleet.
        # -How "dangerous" it is to visit a node in terms of the time required to return to its specific depot.
        enhanced_features = torch.cat([node_features, depot_dist, global_context], dim=1)
        
        current_node = state["current_node"]
        visited_mask = state["visited_mask"].to(self.cfg.device)
        
        # masking: Calculate valid moves
        constraint_mask = self._apply_time_constraints(state, visited_mask)
        
        if constraint_mask.all():
            # If nothing is reachable, return the depot index to end the route gracefully
            truck_obj = next(t for t in self.data["trucks"] if t.id == state["active_truck_id"])
            action = truck_obj.depot_idx
            # Record a "100% probability" log_prob (0.0) so the lists stay the same length
            self.log_probs.append(torch.tensor(0.0, device=self.cfg.device))
            return action
        
        probs = self.policy(enhanced_features, current_node, constraint_mask)        
        # ---  Use probs= instead of logits= ---
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        return action.item()


    def _apply_time_constraints(self, state, visited_mask):
        """
        Calculates a mask where True = Node is unreachable or invalid.
        """
        curr_node = state["current_node"]
        curr_time = state["current_time"]
        active_id = state["active_truck_id"]
        
        truck_obj = next(t for t in self.data["trucks"] if t.id == active_id)
        limit = self.cfg.max_daily_delivery_time_each_truck
        
        mask = visited_mask.clone()
        for i in range(self.data["num_nodes"]):
            if not mask[i]:
                t_to_i = self.data["time_matrix"][curr_node, i].item()
                t_home = self.data["time_matrix"][i, truck_obj.depot_idx].item()
                
                # If this move violates the max_daily_delivery_time_each_truck h wall, hide it from the agent
                if curr_time + t_to_i + t_home > limit:
                    mask[i] = True
        return mask



