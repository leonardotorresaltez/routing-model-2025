import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.encoder import GNNEncoder
from core.models.policy_head import PolicyHead


class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_trucks, num_nodes):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_trucks = num_trucks
        self.num_nodes = num_nodes

        # Your custom GNN encoder (NormalizedGraphConv-based)
        self.encoder = GNNEncoder(input_dim, hidden_dim)

        # Policy + value head
        self.head = PolicyHead(hidden_dim, num_trucks, num_nodes)

    def forward(self, x, edge_index):
    
        # Encode nodes with your GNN
        h = self.encoder(x, edge_index)        # [B, N, H]

        # Policy + value
        logits, value = self.head(h)           # [B, T, N], [B, 1]

        return h, logits, value