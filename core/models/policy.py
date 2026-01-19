import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPolicy(nn.Module):
    def __init__(self, node_dim=2, embed_dim=128):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, nodes, current_node_idx, visited_mask):
        h = self.node_embed(nodes)
        q = self.query(h[current_node_idx])
        k = self.key(h)
        
        scores = torch.matmul(k, q)
        # Masking: visited nodes get -inf score
        scores = scores.masked_fill(visited_mask, float("-inf"))
        
        probs = F.softmax(scores, dim=0)
        return probs