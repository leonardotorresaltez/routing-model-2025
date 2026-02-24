import torch
import torch.nn as nn
from .encoder import GNNEncoder
from .policy_head import PolicyHead

class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_trucks, num_nodes):
        super().__init__()

        self.encoder = GNNEncoder(input_dim, hidden_dim)
        self.actor_head = PolicyHead(hidden_dim, num_trucks, num_nodes)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        _, node_logits, _ = self.actor_head(h)
        mean_pool = torch.mean(h,dim=1)
        max_pool,_ = torch.max(h,dim=1)
        graph_emb = mean_pool+max_pool
        
        value = self.value_head(graph_emb).squeeze(-1)
        
        return None, node_logits, value