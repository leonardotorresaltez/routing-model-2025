import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


    
    
# ----------------------------
# GraphPointer Policy Model
# ----------------------------    
class GraphPointerPolicy(nn.Module):
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

        # node selector pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        
        # truck selector pointer mechanism
        self.truck_query = nn.Linear(embed_dim, embed_dim)
        self.truck_key = nn.Linear(embed_dim, embed_dim)        

    def forward(self, nodes: torch.Tensor , truck_positions: np.array, visited_mask: torch.Tensor):
        """
        nodes:        [N, 2]
        truck_positions: np.array
        visited_mask: [N] bool  (True = forbidden)
        """

        h = self.node_embed(nodes)           # [N, D]
        truck_h = h[truck_positions] 

        visited_mask = visited_mask.bool()
        graph_ctx = self.msg_linear(h.mean(0))  # [D]

  
        
        # ---------- TRUCK SELECTION ----------
                    # [T,D]
        tq = self.truck_query(truck_h + graph_ctx) # [T,D]
        tk = self.truck_key(truck_h)               # [T,D]

        truck_scores = torch.matmul(tq, tk.T).diag()  # [T]
        truck_probs = F.softmax(truck_scores, dim=0)  
        
        # ---------- NODE SELECTION ----------

        query = self.query(truck_h + graph_ctx)  # [D]
        keys = self.key(h)                               # [N, D]

        # Adjust visited_mask to match node_scores dimensions
        node_scores = torch.matmul(query, keys.T)               # [N]       
        node_scores = node_scores.masked_fill(visited_mask.unsqueeze(0), -1e9)           

        # Ensure visited_mask is applied to each row of node_probs
        # expanded_visited_mask = visited_mask.unsqueeze(0).expand(node_scores.size(0), -1)
        expanded_visited_mask = visited_mask.unsqueeze(0)
        node_scores = node_scores.masked_fill(expanded_visited_mask, -1e9)

        # Check if all actions are masked
        if visited_mask.all():
            # -------------------------
            # Add NO-OP action to the scores
            # ------------------------- 
            #noop_score = torch.dot(query, self.noop_key) + self.noop_bias  # scalar
            #noop_score = noop_score.view(1)  # Ensure shape [1]
            #node_scores = torch.cat([node_scores, noop_score], dim=0)  # [N + 1]            

            num_nodes = visited_mask.shape[0] # is this IF case is same as nodes.shape[0]
            # Create a NO-OP action probability
            noop_probs = torch.zeros(num_nodes + 1, device=visited_mask.device)
            # Assign probability 1 to the NO-OP action at index num_nodes
            noop_probs[num_nodes] = 1.0
            node_scores =  noop_probs
        
        node_probs = F.softmax(node_scores, dim=1) # [T,N]

        #truck_dist = torch.distributions.Categorical(truck_probs)
        #truck = truck_dist.sample()
        #node_dist = torch.distributions.Categorical(node_probs[truck])
        #node = node_dist.sample()
        
        #if visited_mask[node]:
        #    print(f"DEBUG: Node {node} is already visited.")        
        
        if self.cfg.debug: print(f"DEBUG: Action probabilities shape: {node_probs.cpu().detach().numpy().shape} | Sum: {node_probs.sum().item():.4f}")
        return truck_probs, node_probs