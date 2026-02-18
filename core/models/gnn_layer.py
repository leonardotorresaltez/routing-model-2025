import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        row, col = edge_index
        
        if x.dim() == 3: 
            B, N, F = x.shape
            agg = torch.zeros_like(x)
            for b in range(B):
                agg[b].index_add_(0, row, x[b, col])
            out = self.linear(agg)
        else: 
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col])
            out = self.linear(agg)

        return torch.relu(out)