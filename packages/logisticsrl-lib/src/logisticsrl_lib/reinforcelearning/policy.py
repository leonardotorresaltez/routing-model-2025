import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# GraphPointer Policy Model
# ----------------------------    
class GraphPointerPolicy(nn.Module):
    def __init__(self, cfg, node_dim=3, embed_dim=128):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim  # Save for the attention scaling factor
        
        # Node embedding
        self.node_embed = nn.Linear(node_dim, embed_dim)

        # -------------------------------------------------
        # UPGRADE: Attention Context (Replaces msg_linear)
        # -------------------------------------------------
        self.ctx_query = nn.Linear(embed_dim, embed_dim)
        self.ctx_key = nn.Linear(embed_dim, embed_dim)
        self.ctx_value = nn.Linear(embed_dim, embed_dim)

        # Pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, nodes: torch.Tensor , current_node: int, visited_mask: torch.Tensor):
        """
        nodes:        [N, node_dim]
        current_node: int
        visited_mask: [N] bool  (True = forbidden)
        """
        visited_mask = visited_mask.bool()

        # 1. Embed the nodes
        h = self.node_embed(nodes)           # [N, D]
        
        # -------------------------------------------------
        # 2. Compute Dynamic Graph Context (Attention)
        # -------------------------------------------------
        # "Where am I currently?" -> Query
        curr_q = self.ctx_query(h[current_node])  # [D]
        
        # "What does the map look like?" -> Keys and Values
        ctx_k = self.ctx_key(h)                   # [N, D]
        ctx_v = self.ctx_value(h)                 # [N, D]
        
        # Calculate Scaled Dot-Product Attention: (K @ Q) / sqrt(D)
        ctx_scores = torch.matmul(ctx_k, curr_q) / math.sqrt(self.embed_dim)  # [N]
        
        # CRITICAL: Mask out visited nodes so they don't corrupt our context
        ctx_scores = ctx_scores.masked_fill(visited_mask, -1e9)
        
        # Convert to percentages (weights)
        ctx_weights = F.softmax(ctx_scores, dim=0)  # [N]
        
        # Multiply weights by values to get our final context vector
        graph_ctx = torch.matmul(ctx_weights, ctx_v)  # [D]

        # -------------------------------------------------
        # 3. Final Pointer Mechanism
        # -------------------------------------------------
        # Combine the current node with the new dynamic context
        query = self.query(h[current_node] + graph_ctx)  # [D]
        keys = self.key(h)                               # [N, D]

        scores = torch.matmul(keys, query)               # [N]
        
        scores = scores.masked_fill(visited_mask, -1e9)

        probs = F.softmax(scores, dim=0)
        
        if self.cfg.debug: 
            print(f"DEBUG: Action probabilities shape: {probs.cpu().detach().numpy().shape} | Sum: {probs.sum().item():.4f}")
            
        return probs
    

class GraphPointerPolicy_old(nn.Module):
    # This implemented context as a simple mean of node embeddings, the new version has upgraded it to a proper attention mechanism
    def __init__(self, cfg, node_dim=3, embed_dim=128):
        super().__init__()
        self.cfg = cfg
        # -------------------------
        # Añadir NO-OP
        # ------------------------- 
        self.noop_key = nn.Parameter(torch.randn(embed_dim))
        self.noop_bias = nn.Parameter(torch.zeros(1))
        
        # Node embedding
        self.node_embed = nn.Linear(node_dim, embed_dim)

        # Simple graph message passing (1 step)
        self.msg_linear = nn.Linear(embed_dim, embed_dim)

        # Pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, nodes: torch.Tensor , current_node: int, visited_mask: torch.Tensor):
        """
        nodes:        [N, 2]
        current_node: int
        visited_mask: [N] bool  (True = forbidden)
        """

        visited_mask = visited_mask.bool()

        h = self.node_embed(nodes)           # [N, D]
        graph_ctx = self.msg_linear(h.mean(0))  # [D]

        query = self.query(h[current_node] + graph_ctx)  # [D]
        keys = self.key(h)                               # [N, D]

        scores = torch.matmul(keys, query)               # [N]
        
        scores = scores.masked_fill(visited_mask, -1e9)
        


        # Check if all actions are masked
        if visited_mask.all():
            # -------------------------
            # Add NO-OP action to the scores
            # ------------------------- 
            noop_score = torch.dot(query, self.noop_key) + self.noop_bias  # scalar
            noop_score = noop_score.view(1)  # Ensure shape [1]
            scores = torch.cat([scores, noop_score], dim=0)  # [N + 1]            
            
            num_nodes = visited_mask.shape[0] # is this IF case is same as nodes.shape[0]
            # Create a NO-OP action probability
            noop_probs = torch.zeros(num_nodes + 1, device=visited_mask.device)
            # Assign probability 1 to the NO-OP action at index num_nodes
            noop_probs[num_nodes] = 1.0
            return noop_probs

        probs = F.softmax(scores, dim=0)
        if self.cfg.debug: print(f"DEBUG: Action probabilities shape: {probs.cpu().detach().numpy().shape} | Sum: {probs.sum().item():.4f}")
        return probs
    
    

# ----------------------------
# FactorizedFleetPolicy Policy Model
# ----------------------------    
class FactorizedFleetPolicy(nn.Module):
    #_SUM_OTHER_DIM = 2
    
    def __init__(self, cfg, input_features_size=10, embed_dim=128):
        super().__init__()
        self.cfg = cfg

        self.embed_dim = embed_dim
        # Feature embedding — initialized at construction so the optimizer tracks it
        self.features = nn.Linear(input_features_size, embed_dim)
        #self.features = None
        # Simple graph message passing (1 step)
        self.msg_linear = nn.Linear(embed_dim, embed_dim)

        # node selector pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

        # truck selector pointer mechanism
        self.truck_query = nn.Linear(embed_dim, embed_dim)
        self.truck_key = nn.Linear(embed_dim, embed_dim)

        # per-truck time projection: injects normalized elapsed time into truck embedding
        # so trucks at same position are distinguishable when their remaining time differs
        self.time_proj = nn.Linear(1, embed_dim)

        # critic: value head V(s)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )        

    def forward(self, observation_space_as_features: torch.Tensor, truck_positions: np.array, visited_mask: torch.Tensor, inactive_trucks_mask: torch.Tensor, truck_times: torch.Tensor = None):
        """
        observation_space_as_features: [N, size_dim]
        truck_positions: np.array
        visited_mask: [T, N] bool  (True = forbidden)
        inactive_trucks_mask: [T] bool (True = inactive)
        truck_times: [T] float, normalized elapsed time per truck in [0, 1]
        """

        # Dynamically initialize self.features if it is None
        #if self.features is None:
        #    input_features_size = observation_space_as_features.size(1)  # number of nodes dynamically
        #    self.features = nn.Linear(input_features_size, self.embed_dim)

        h = self.features(observation_space_as_features)           # [N, D]
        truck_h = h[truck_positions]                               # [T, D]

        # Inject per-truck elapsed time so trucks at the same depot are distinguishable.
        # time_proj maps scalar elapsed_time → [D], added to each truck's position embedding.
        if truck_times is not None:
            truck_h = truck_h + self.time_proj(truck_times.unsqueeze(1))  # [T, D]

        visited_mask = visited_mask.bool()
        graph_ctx = self.msg_linear(h.mean(0))  # [D]

        # ---------- TRUCK SELECTION ----------
        tq = self.truck_query(truck_h + graph_ctx) # [T,D]
        tk = self.truck_key(truck_h)               # [T,D]

        truck_scores = torch.matmul(tq, tk.T).diag()  # [T]
        truck_scores = truck_scores.masked_fill(inactive_trucks_mask, -1e9)  # Apply truck time mask
        truck_probs = F.softmax(truck_scores, dim=0)  
        
        # ---------- NODE SELECTION ----------
        query = self.query(truck_h + graph_ctx)  # [D]
        keys = self.key(h)                               # [N, D]

        node_scores = torch.matmul(query, keys.T)               # [N]       
        node_scores = node_scores.masked_fill(visited_mask, -1e9)    
        node_probs = F.softmax(node_scores, dim=-1)  # Corrected dimension for softmax to apply along the last axis     
 
        
        # critic: V(s) from graph context
        value = self.value_head(graph_ctx).squeeze(-1)  # scalar

        if self.cfg.debug: print(f"DEBUG: Action probabilities shape: {node_probs.cpu().detach().numpy().shape} | Sum: {node_probs.sum().item():.4f}")
        return truck_probs, node_probs, value