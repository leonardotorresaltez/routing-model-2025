import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GraphEncoder', 'GraphPointer', 'GraphPointerNetwork']

# ---------------------------
# Simple Graph Encoder (GNN-like)
# ---------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)

    def forward(self, x, adj):
        """
        x: [N, in_dim]      node features
        adj: [N, N]        adjacency matrix (0/1)
        """
        h = self.linear(x)             # Line 21: Feature transformation (X·W) -- [N, hidden_dim]
        h = torch.matmul(adj, h)       # Line 22: Feature propagation (A·h) -- message passing
        return F.relu(h)               # Line 23: Activation σ(·)
    
# ---------------------------
# Pointer Network over Graph Nodes
# ---------------------------
class GraphPointer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_embeddings):
        """
        node_embeddings: [N, hidden_dim]
        """
        keys = self.key(node_embeddings)           # [N, hidden_dim]
        scores = torch.matmul(keys, self.query)    # [N]
        return F.softmax(scores, dim=0)

# ---------------------------
# Full Graph Pointer Network
# ---------------------------
class GraphPointerNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim)
        self.pointer = GraphPointer(hidden_dim)

    def forward(self, x, adj):
        h = self.encoder(x, adj)
        probs = self.pointer(h)
        return probs
    