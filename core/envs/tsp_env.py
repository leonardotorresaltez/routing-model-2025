import random
import sys

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from core.utils.routing import apply_2opt, calculate_route_time


class TSPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.num_nodes = cfg.num_nodes
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(0.0, 1.0, (self.num_nodes, 2), dtype=np.float32), # TODO: Fixed nodes from a csv
            # TODO: Take adjacency distance matrix from a csv
            "current": spaces.Discrete(self.num_nodes),
            "visited": spaces.MultiBinary(self.num_nodes)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.nodes = torch.rand(self.num_nodes, 2)
        self.current = random.randrange(self.num_nodes)
        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.visited[self.current] = True
        self.tour = [self.current]
        return self._get_state(), {}

    def _get_state(self):
        return {
            "nodes": self.nodes.clone(),
            "current": self.current,
            "visited": self.visited.clone()
        }

    def step(self, action):
        prev = self.current
        self.current = action
        self.visited[action] = True
        self.tour.append(action)

        dist = torch.norm(self.nodes[prev] - self.nodes[action])
        reward = -dist # Minimize distance = Maximize negative distance

        terminated = self.visited.all()
        return self._get_state(), reward, terminated, False, {}
    
    
    




class MDVRPEnv(gym.Env):
    """ 
    Summary: It contains the "map" (locations of depots and customers) and the "rules" (trucks must start at their depot and visit customers).
    
    State: It tells the Agent which nodes exist and which ones have already been visited.
    Reward: It calculates the "score." Since the goal is to be efficient, it gives a negative reward based on total travel time (shorter routes = less negative = better score).
    
    
    single-step reward calculation based on the total travel time of all 50 trucks.    
    Single-Step Journey: It calculates all 50 routes at once (on a for loop) and returns terminated=True.
    Multi-Depot Return: Each truck returns to its specific depot_idx.
    """
    def __init__(self, cfg, data):
        super().__init__()
        self.cfg = cfg
        # We use the time-proximity profiles as observations
        self.node_features = data["node_features"]
        self.time_matrix = data["time_matrix"]
        self.trucks = data["trucks"]
        self.depots = data["depots"]
        self.customers = data["customers"]
        self.num_nodes = data["num_nodes"]

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(0.0, 1.0, (self.num_nodes, self.num_nodes), dtype=np.float32)
        })
        
        # In a one-shot environment, the action space is handled logically by the agent
        self.action_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        """
        The environment is the "judge" that calculates how well the agent performed.
        
        action: Dict[truck_id, List[customer_idx]]
        """
        total_time = 0.0
        visited_customers = set()
        
        # ONESHOT: one step is an entire truck loop
        for truck in self.trucks:
            route = action.get(truck.id, [])
            if not route:
                continue
            
            prev_idx = truck.depot_idx
            truck_time = 0.0
            
            for cust_idx in route:
                # Add time from the CSV-based time_matrix
                truck_time += self.time_matrix[prev_idx, cust_idx].item()
                visited_customers.add(cust_idx)
                prev_idx = cust_idx
            
            # Add time to return to the depot
            truck_time += self.time_matrix[prev_idx, truck.depot_idx].item()
            total_time += truck_time

        # Negative reward to minimize total time
        reward = -total_time
        
        # Penalty for skipping customers
        missing = len(self.customers) - len(visited_customers)
        reward -= missing * 50.0 # High penalty to ensure coverage
        
        # ONESHOT: always returns terminated=True
        # ONESHOT: The agent provides the whole day's plan, the environment calculates the whole day's reward, and the episode ends.
        return self._get_obs(), reward, True, False, {"total_time": total_time}

    def _get_obs(self): 
        """
        The environment is responsible for telling the agent what the world looks like.
        """
        return {"node_features": self.node_features.clone()}




class MDVRP_one_agent_per_truck_env(gym.Env):
    def __init__(self, cfg, data, truck_starts):
        super().__init__()
        self.cfg = cfg
        self.node_features = data["node_features"]
        self.time_matrix = data["time_matrix"]
        self.trucks = data["trucks"]
        self.depots = data["depots"]
        self.customers = data["customers"]
        self.num_nodes = data["num_nodes"]
        self.truck_starts = truck_starts

        # Define the ID mapping BEFORE calling reset()
        self.truck_id_to_idx = {t.id: i for i, t in enumerate(self.trucks)}
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(0.0, 1.0, (self.num_nodes, self.num_nodes), dtype=np.float32)
        })        
        self.action_space = spaces.Discrete(self.num_nodes)        
        self.use_2opt = False
        
        # Safe to call now
        self.reset()
        
    def _get_info(self, terminated):
        """
        Helper to create the info dictionary used for logging and debugging.
        """
        info = {}
        if terminated:
            # Full report at the end of the day
            info["truck_results"] = {
                tid: {
                    "route": data["route"],
                    "time": data["ready_time"]
                } for tid, data in self.truck_states.items()
            }
            # Total time spent by the entire fleet
            info["total_time"] = sum(t["ready_time"] for t in self.truck_states.values())
            
            # Count only actual customers visited (exclude depots)
            # We subtract the number of depots because depots were pre-masked as 'visited'
            info["total_visited"] = int(self.visited_mask.sum().item() - len(self.depots))
            
        return info



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Track state for each truck
        self.truck_states = {
            t.id: {
                "current_node": t.depot_idx,
                "ready_time": 0.0,
                "route": []
            } for t in self.trucks
        }
        self.visited_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        # Pre-mask depots so they aren't visited as customers
        for d in self.depots:
            self.visited_mask[d.idx] = True
        
        self.truck_active = {t.id: True for t in self.trucks}            
       
        
        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.current_positions = self.truck_starts.copy()
        
        # Mark all depots as visited for all trucks
        for pos in self.truck_starts:
            self.visited[pos] = True
        
        self.tours = [[pos] for pos in self.truck_starts]       
        
               
        return self._get_obs(), {}

    def step(self, action):
        truck_id = self._get_next_truck_id()
        truck_state = self.truck_states[truck_id]
        truck_obj = next(t for t in self.trucks if t.id == truck_id)
        
        
        prev_node = truck_state["current_node"]
        travel_time = self.time_matrix[prev_node, action].item()
        # Time from the POTENTIAL new node back to depot
        time_home_from_action = self.time_matrix[action, truck_obj.depot_idx].item()
        
        # THE HARD ENFORCEMENT
        limit = self.cfg.max_daily_delivery_time_each_truck
        total_time_if_visited = truck_state["ready_time"] + travel_time + time_home_from_action
        
        if total_time_if_visited <= limit:
            # VALID MOVE: Update state
            truck_state["current_node"] = action
            truck_state["ready_time"] += travel_time
            truck_state["route"].append(action)
            self.visited_mask[action] = True
            # Goal: Minimize time (-travel_time)
            reward = - travel_time # FIXME
            self.visited[action] = True # same as visited_mask FIXME
            self.tours[self.truck_id_to_idx[truck_id]].append(action)
        else: # safety code, it should NEVER be reached out
            # INVALID MOVE: This truck's day ends at its CURRENT location
            print('holaaaaaaaa')
            sys.exit()
            time_home_from_prev = self.time_matrix[prev_node, truck_obj.depot_idx].item()
            truck_state["ready_time"] += time_home_from_prev
            truck_state["current_node"] = truck_obj.depot_idx
            truck_state["route"].append(truck_obj.depot_idx)
            
            # RETIRE THE TRUCK
            self.truck_active[truck_id] = False
            reward = -time_home_from_prev - 50.0 # Heavy penalty for invalid choice
            
        # Check if truck should be retired anyway (no more FUTURE moves possible)
        if self.truck_active[truck_id]:
            if not self._has_valid_next_move(truck_id):
                # Only append depot if we aren't already there
                if truck_state["current_node"] != truck_obj.depot_idx:
                    h_time = self.time_matrix[truck_state["current_node"], truck_obj.depot_idx].item()
                    truck_state["ready_time"] += h_time
                    truck_state["current_node"] = truck_obj.depot_idx
                    truck_state["route"].append(truck_obj.depot_idx)
    
                self.truck_active[truck_id] = False

        terminated = self.visited_mask.all() or not any(self.truck_active.values())
        info = self._get_info(terminated)
        
        if terminated:
            raw_total_time = info["total_time"]
            optimized_total_time = raw_total_time
            
            # keep track of the raw time for the reward signal
            raw_total_time = info["total_time"]
            optimized_total_time = raw_total_time
            
            # apply 2-opt if requested (Last Episode)
            if self.use_2opt:
                print('raw_total_time ', raw_total_time)
                print('hereeeeeeeeeeeeeeee')
                optimized_total_time = 0.0
                for tid, res in info["truck_results"].items():
                    if res["route"]:
                        truck_obj = next(t for t in self.trucks if t.id == tid)
                        # Apply local search
                        res["route"] = apply_2opt(res["route"], truck_obj.depot_idx, self.time_matrix)
                        res["time"] = calculate_route_time(res["route"], truck_obj.depot_idx, self.time_matrix)
                    optimized_total_time += res.get("time", 0.0)
                print('optimized_total_time ', optimized_total_time)
            
            # Add optimized time to info for logging
            info["optimized_total_time"] = optimized_total_time
            
            unvisited_count = (self.visited_mask == False).sum().item()
            reward -= (unvisited_count * 500.0) # Heavy penalty # Goal: maximize clients
        return self._get_obs(), reward, terminated, False, info

    def _has_valid_next_move(self, truck_id):
        """Helper to check if a truck has at least one reachable unvisited customer."""
        state = self.truck_states[truck_id]
        truck_obj = next(t for t in self.trucks if t.id == truck_id)
        limit = self.cfg.max_daily_delivery_time_each_truck
        
        for i in range(self.num_nodes):
            if not self.visited_mask[i]:
                t_to_i = self.time_matrix[state["current_node"], i].item()
                t_home = self.time_matrix[i, truck_obj.depot_idx].item()
                if state["ready_time"] + t_to_i + t_home <= limit:
                    return True
        return False




    def _get_next_truck_id(self):
        # Only select from trucks that are still allowed to work
        available = [tid for tid, active in self.truck_active.items() if active]
        if not available: return None
        return min(available, key=lambda tid: self.truck_states[tid]["ready_time"]) # The truck selected is the one with less time of travel


    def _get_obs(self):
        truck_id = self._get_next_truck_id()
        if truck_id is None:
            return {"active_truck_id": None}
        
        
        # Find the number of trucks dynamically
        num_trucks = len(self.trucks)
        
        # idx: Converts the truck's unique ID (which might be a string or a non-sequential number) into a 0-based integer. This index is necessary for accessing positions in tensors and arrays.
        # truck_identity: A One-Hot Encoded vector (e.g., [0, 0, 1, 0, ...]). It tells the neural network: "You are currently making a decision for Truck #X." Without this, the model wouldn't know which truck it is controlling.
        # truck_state: Retrieves the dynamic status of the active truck from the environment. It tracks where the truck is currently located (current_node) and how many hours it has worked so far (ready_time).
        # truck_obj: Retrieves the static properties of the truck (defined in your data files). This is primarily used to find the truck's depot_idx, so the model can calculate how far away "home" is from any given customer.
        # TRUCK IDENTITY (indexed)
        truck_identity = torch.zeros(num_trucks)
        idx = self.truck_id_to_idx[truck_id] # Map the data ID to a 0-based vector index
        truck_identity[idx] = 1.0
        truck_state = self.truck_states[truck_id]
        truck_obj = next(t for t in self.trucks if t.id == truck_id)

        # GLOBAL FLEET STATUS
        limit = self.cfg.max_daily_delivery_time_each_truck
        fleet_ready_times = torch.tensor([
            self.truck_states[t.id]["ready_time"] / limit 
            for t in self.trucks
        ], dtype=torch.float32)
        


        # EPOT PROXIMITY (Fixed for Tensor types)
        # Access the column directly from the matrix and ensure it is a float tensor
        depot_dist = self.time_matrix[:, truck_obj.depot_idx].clone().detach().float()

        return {
            "node_features": self.node_features.clone(),
            "current_node": truck_state["current_node"],
            "current_time": truck_state["ready_time"],
            "active_truck_id": truck_id,
            "visited_mask": self.visited_mask.clone(),
            "depot_dist": depot_dist,
            "fleet_status": fleet_ready_times,
            "truck_identity": truck_identity
        }

