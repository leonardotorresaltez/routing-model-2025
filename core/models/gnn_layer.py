import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.neighbor_linear = nn.Linear(in_features, out_features)
        self.self_linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # x shape: [Batch, Num_Nodes, Features] or [Num_Nodes, Features]
        # edge_index shape: [2, Num_Edges]
        
        row, col = edge_index
        
        if x.dim() == 3:
            # Optimized Batch Message Passing
            B, N, F_in = x.shape
            # x[B, col] gathers neighbor features
            # We use index_add on the node dimension (dim=1)
            agg = torch.zeros(B, N, F_in, device=x.device)
            
            # This ensures only connected neighbors are summed
            # x.gather is often used here, but for MDVRP, index_add is reliable
            neighbor_features = x[:, col, :] # Gather neighbor states
            agg.index_add_(1, row, neighbor_features) 
            
            out = self.neighbor_linear(agg) + self.self_linear(x)
        else:
            # Single instance mode (Standard)
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col])
            out = self.neighbor_linear(agg) + self.self_linear(x)

        return F.relu(out)