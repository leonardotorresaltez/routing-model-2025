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
        
        
        
        
        
        #  The Shared GNN Embedding: both the Actor and the Critic share the initial node embedding layer
        self.node_embed = nn.Linear(node_dim, embed_dim)




        # -------------------------------------------------
        # The Policy Network (The Actor)
        # -------------------------------------------------
        # The Actor's job is to look at the map, look at where the trucks currently are, and output a probability distribution over which customer each truck should visit next.
        #  PPO uses this to pick actions and calculate the probability Ratio
        self.ctx_query = nn.Linear(embed_dim, embed_dim)
        self.ctx_key = nn.Linear(embed_dim, embed_dim)
        self.ctx_value = nn.Linear(embed_dim, embed_dim)

        # Pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        
        
        
    
        
        self.noop_key = nn.Parameter(torch.randn(embed_dim))
        self.noop_bias = nn.Parameter(torch.tensor(0.0))  # Changed
        
        
        
        
        # ---------------------------------
        # The Value Network (The Critic)
        # ---------------------------------
        # How much total reward do I expect to get from here until the end of the episode
        # PPO uses this to calculate the Advantage (did the Actor do better or worse than the Critic expected?)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )





    def forward(self, nodes: torch.Tensor , current_trucks: torch.Tensor, visited_mask: torch.Tensor):
        """
        nodes: [N, 2] - node coordinates
        action_mask: [T*N] - valid (truck, customer) pairs, - 1 = invalid, 0 = valid
        
        Returns: probs [T*N]
        """
        
     
        visited_mask = visited_mask.bool()

        # Embed the nodes
        h = self.node_embed(nodes)           # [N, D]
        
        # Extract specific embeddings for where each truck currently is
        truck_h = h[current_trucks]          # [T, D]
        
        
        

        
        
        
        
        
        # -------------------------------------------------
        # Compute Dynamic Graph Context (Attention)
        # -------------------------------------------------
        # "Where am I currently?" -> Query for EACH truck, Where are my trucks right now?
        curr_q = self.ctx_query(truck_h)     # [T, D]        
        # "What does the map look like?" -> Keys and Values
        ctx_k = self.ctx_key(h)              # [N, D] # What are the features of all nodes on the map
        ctx_v = self.ctx_value(h)            # [N, D] # What information do I pull from those node
        
        # Calculate Scaled Dot-Product Attention: (K @ Q) / sqrt(D)
        # ctx_scores = torch.matmul(ctx_k, curr_q) / math.sqrt(self.embed_dim)  # [N]
        ctx_scores = torch.matmul(curr_q, ctx_k.transpose(0, 1)) / math.sqrt(self.embed_dim)

        
        # Convert to percentages (weights)
        # ctx_weights = F.softmax(ctx_scores, dim=0)  # [N]
        ctx_weights = F.softmax(ctx_scores, dim=1)  # [T, N]
        
        # Multiply weights by values to get our final context vector
        # graph_ctx = torch.matmul(ctx_weights, ctx_v)  # [D]
        graph_ctx = torch.matmul(ctx_weights, ctx_v)  # [T, D]

        # -------------------------------------------------
        # Final Pointer Mechanism
        # -------------------------------------------------
        query = self.query(graph_ctx + truck_h)
        keys = self.key(h)                   # [N, D]
        scores = torch.matmul(query, keys.transpose(0, 1))  # [T, N]
        scores = scores.view(-1)  # [T*N]
        scores = scores.masked_fill(visited_mask, -1e9)  # Set invalid actions to -infinity
        # print(f"DEBUG: scores shape={scores.shape}, visited_mask shape={visited_mask.shape}")
        # print(f"DEBUG: num masked actions = {visited_mask.sum()}")
        
        # Convert mask to boolean and apply: True where invalid
        
        
        # -------------------
        # Critic value - calculation
         # -------------------
        # graph_embed = h.mean(dim=0)          # [D]
        # value = self.value_head(graph_embed) # [1]
        # give the Critic the rich attention context AND the truck locations
        fleet_context = graph_ctx.mean(dim=0) + truck_h.mean(dim=0)  # [D]
        value = self.value_head(fleet_context)  
        

        probs = F.softmax(scores, dim=0) # [T*N]
        
        # probs = probs + 1e-8                 # Prevent pure zeros
        # probs = probs / probs.sum()          # Re-normalize
        
        # if self.cfg.debug: 
        #     print(f"DEBUG: Action probabilities shape: {probs.cpu().detach().numpy().shape} | Sum: {probs.sum().item():.4f}")
            
        return probs, value
    