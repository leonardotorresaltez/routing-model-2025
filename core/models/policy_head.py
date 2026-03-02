import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, num_trucks: int, num_nodes: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_trucks = num_trucks
        self.num_nodes = num_nodes

        # Per-truck query embeddings (learned)
        self.truck_query = nn.Parameter(torch.randn(num_trucks, hidden_dim))

        # Optional projection for stability
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # Value head (critic)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [B, N, H] node embeddings from GNN encoder

        Returns:
            logits: [B, T, N] per-truck logits over nodes
            value:  [B, 1]    state value
        """
        B, N, H = h.shape

        # Normalize and project node embeddings
        h = F.layer_norm(h, (H,))
        h = F.relu(self.proj(h))  # [B, N, H]

        # Normalize truck queries
        queries = F.normalize(self.truck_query, dim=-1)          # [T, H]
        queries = queries.unsqueeze(0).unsqueeze(2)              # [1, T, 1, H]

        # Normalize node keys
        keys = F.normalize(h, dim=-1).unsqueeze(1)               # [B, 1, N, H]

        # Scaled dot-product attention: per-truck logits over nodes
        logits = (queries * keys).sum(dim=-1) / (H ** 0.5)       # [B, T, N]

        # State value from mean-pooled node embeddings
        value = self.value_head(h.mean(dim=1))                   # [B, 1]

        return logits, value