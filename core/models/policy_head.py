import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, num_trucks, num_nodes):
        super().__init__()
        self.truck_head = nn.Linear(hidden_dim, num_trucks)
        self.node_head  = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        if h.dim() == 3: # Batch mode [B, N, hidden]
            graph_emb = h.mean(dim=1) # Mean across nodes, keep batch
            truck_logits = self.truck_head(graph_emb) 
            node_logits = self.node_head(h).squeeze(-1) # [B, N]
        else: # Single step mode [N, hidden]
            graph_emb = h.mean(dim=0)
            truck_logits = self.truck_head(graph_emb)
            node_logits = self.node_head(h).squeeze(-1) # [N]
            
        return truck_logits, node_logits, graph_emb