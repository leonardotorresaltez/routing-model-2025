import torch
import torch.nn as nn
import torch.nn.functional as F

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
        h = self.linear(x)             # [N, hidden_dim]
        h = torch.matmul(adj, h)       # message passing
        return F.relu(h)
    
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

# ---------------------------
#  example
# ---------------------------
N = 4  # number of nodes
in_dim = 3
hidden_dim = 8

# Node features
x = torch.randn(N, in_dim)
print("Input features:", x)

# Adjacency matrix (simple graph)
adj = torch.tensor([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1]
], dtype=torch.float32)

model = GraphPointerNetwork(in_dim, hidden_dim)

probs = model(x, adj)
selected_node = torch.argmax(probs).item()

print("Pointer probabilities:", probs)
print("Selected node:", selected_node)    