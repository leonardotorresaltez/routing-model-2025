import torch
import torch.nn as nn

class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, num_trucks, num_nodes):
        super().__init__()
        self.truck_head = nn.Linear(hidden_dim, num_trucks)
        self.node_head  = nn.Linear(hidden_dim, 1)  # FIXED

    def forward(self, h):
        # Graph embedding
        graph_emb = h.mean(dim=0)
        # Truck logits
        truck_logits = self.truck_head(graph_emb)
        # Node logits (one score per node)
        node_logits = self.node_head(h).squeeze(-1)  # (num_nodes,)
        return truck_logits, node_logits, graph_emb