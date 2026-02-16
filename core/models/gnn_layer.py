import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        row, col = edge_index  # row = target, col = source

        # Aggregate neighbor features
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])

        # Apply linear transform
        out = self.linear(agg)
        return F.relu(out)