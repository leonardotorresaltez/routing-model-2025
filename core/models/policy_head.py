import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, num_trucks, num_nodes):
        super().__init__()
        # Project node embeddings to a score for each specific truck
        self.node_head = nn.Linear(hidden_dim, num_trucks)
        
    def forward(self, h):
        """
        h: Node embeddings from GNN [Batch, Num_Nodes, Hidden] 
           or [Num_Nodes, Hidden]
        """
        if h.dim() == 3: # Batch mode [B, N, hidden]
            # 1. Global context for the Critic
            graph_emb = h.mean(dim=1) 
            
            # 2. Generate Joint Logits: [B, N, num_trucks]
            # Each node gets a score per truck
            node_scores = self.node_head(h) 
            
            # 3. Transpose to [Batch, num_trucks, num_nodes]
            # This allows us to sample a destination for every truck
            node_logits = node_scores.transpose(1, 2)
            
        else: # Single step (inference/rollout) [N, hidden]
            graph_emb = h.mean(dim=0)
            node_scores = self.node_head(h) # [N, num_trucks]
            node_logits = node_scores.t() # [num_trucks, num_nodes]
            
     
        return None, node_logits, graph_emb