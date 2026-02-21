import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# GraphPointer Policy Model
# ----------------------------    
class GraphPointerPolicy(nn.Module):
    def __init__(self, cfg, node_dim=2, embed_dim=128): # nodes are 2D (x, y)
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
        
        self.noop_key = nn.Parameter(torch.randn(embed_dim))
        self.noop_bias = nn.Parameter(torch.tensor(0.0))  # Changed

    def forward(self, nodes: torch.Tensor , visited_mask: torch.Tensor):
        """
        nodes: [N, 2] - node coordinates
        action_mask: [T*N] - valid (truck, customer) pairs
        
        Returns: probs [T*N]
        """
        visited_mask = visited_mask.bool()

        # 1. Embed the nodes
        h = self.node_embed(nodes)           # [N, D]
        
        # -------------------------------------------------
        # 2. Compute Dynamic Graph Context (Attention)
        # -------------------------------------------------
        # "Where am I currently?" -> Query
        # curr_q = self.ctx_query(h[current_node])  # [D]
        
        # "What does the map look like?" -> Keys and Values
        ctx_q = self.ctx_query(h.mean(0))    # [D] - global query
        ctx_k = self.ctx_key(h)              # [N, D]
        ctx_v = self.ctx_value(h)            # [N, D]
        
        # Calculate Scaled Dot-Product Attention: (K @ Q) / sqrt(D)
        ctx_scores = torch.matmul(ctx_k, ctx_q) / math.sqrt(self.embed_dim)  # [N]
        
        # Convert to percentages (weights)
        ctx_weights = F.softmax(ctx_scores, dim=0)  # [N]
        
        # Multiply weights by values to get our final context vector
        graph_ctx = torch.matmul(ctx_weights, ctx_v)  # [D]

        # -------------------------------------------------
        # 3. Final Pointer Mechanism
        # -------------------------------------------------
        # Combine the current node with the new dynamic context
        query = self.query(graph_ctx)        # [D]
        keys = self.key(h)                   # [N, D]
        scores = torch.matmul(keys, query)   # [N]

        scores = torch.matmul(keys, query)               # [N]
        
        # Replicate scores for each truck
        num_nodes = scores.shape[0]
        num_trucks = visited_mask.shape[0] // num_nodes
        scores = scores.repeat(num_trucks)   # [T*N]
        
        # CRITICAL: Mask out visited nodes so they don't corrupt our context
        scores = scores.masked_fill(visited_mask, -1e9) # Now [N+1] matches visited_mask [N+1]

        probs = F.softmax(scores, dim=0) # [T*N]
        
        if self.cfg.debug: 
            print(f"DEBUG: Action probabilities shape: {probs.cpu().detach().numpy().shape} | Sum: {probs.sum().item():.4f}")
            
        return probs
    