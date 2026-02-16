import torch.nn as nn
from .gnn_layer import GraphConv

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        return h