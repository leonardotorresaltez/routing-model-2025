import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.neighbor_linear = nn.Linear(in_features, out_features)
        self.self_linear = nn.Linear(in_features, out_features)

    # core/models/gnn_layer.py

    def forward(self, x, edge_index):
        row, col = edge_index  # row = target, col = source

        if x.dim() == 3:
            B, N, F_in = x.shape
            neigh = x[:, col, :]  # [B, E, F]

            # 1. Prepare Degree Matrix [B, N]
            deg = torch.zeros(B, N, device=x.device)
            
            # Create a source tensor of ones shaped [1, E] to match deg's rank
            # This allows index_add_ to work on dim 1 across the batch
            source_ones = torch.ones((1, row.size(0)), dtype=torch.float32, device=x.device)
            
            # Use index_add_ on dim 1
            deg.index_add_(1, row, source_ones.expand(B, -1))
            deg = deg.clamp(min=1.0)

            # 2. Normalize
            norm = 1.0 / torch.sqrt(deg[:, row] * deg[:, col])  # [B, E]
            
            # 3. Aggregate [B, N, F_in]
            agg = torch.zeros(B, N, F_in, device=x.device)
            # neigh * norm.unsqueeze(-1) is [B, E, F_in]
            # row is [E], so this adds to node indices across all batches
            agg.index_add_(1, row, neigh * norm.unsqueeze(-1))
            
        else:
            # Single graph logic (Keep as is, but ensure agg initialization is correct)
            N, F_in = x.shape
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
            deg = deg.clamp(min=1.0)

            norm = 1.0 / torch.sqrt(deg[row] * deg[col])
            agg = torch.zeros(N, F_in, device=x.device)
            agg.index_add_(0, row, x[col] * norm.unsqueeze(-1))

        out = self.neighbor_linear(agg) + self.self_linear(x)
        return F.relu(out)