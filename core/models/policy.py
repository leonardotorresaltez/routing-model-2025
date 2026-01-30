import torch
import torch.nn as nn
import torch.nn.functional as F


    
    
# ----------------------------
# GraphPointer Policy Model
# ----------------------------    
class GraphPointerPolicy(nn.Module):
    def __init__(self, node_dim=2, embed_dim=128):
        super().__init__()

        # Node embedding
        self.node_embed = nn.Linear(node_dim, embed_dim)

        # Simple graph message passing (1 step)
        self.msg_linear = nn.Linear(embed_dim, embed_dim)

        # Pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, nodes, current_node, visited_mask):
        """
        nodes:        [N, 2]
        current_node: int
        visited_mask: [N] bool  (True = forbidden)
        """

        visited_mask = visited_mask.bool()

        h = self.node_embed(nodes)           # [N, D]
        graph_ctx = self.msg_linear(h.mean(0))  # [D]

        query = self.query(h[current_node] + graph_ctx)  # [D]
        keys = self.key(h)                               # [N, D]

        scores = torch.matmul(keys, query)               # [N]
        scores = scores.masked_fill(visited_mask, -1e9)

        probs = F.softmax(scores, dim=0)
        return probs    