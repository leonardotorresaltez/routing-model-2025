import torch.nn as nn
from core.models.gnn_layer import NormalizedGraphConv
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.conv1 = NormalizedGraphConv(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.conv2 = NormalizedGraphConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.ln1(h)
        h = F.relu(h)              # <-- IMPORTANT
        h = self.dropout(h)

        # Layer 2 + residual
        h2 = self.conv2(h, edge_index)
        h2 = self.ln2(h2)
        h = h + h2                 # residual
        h = F.relu(h)              # <-- IMPORTANT
        h = self.dropout(h)

        return h