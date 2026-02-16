import torch
import torch.nn as nn
from .encoder import GNNEncoder
from .policy_head import PolicyHead

class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_trucks, num_nodes):
        super().__init__()

        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.actor_head = PolicyHead(hidden_dim, num_trucks, num_nodes)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        truck_logits, node_logits, graph_emb = self.actor_head(h)
        value = self.value_head(graph_emb).squeeze(-1)
        return truck_logits, node_logits, value