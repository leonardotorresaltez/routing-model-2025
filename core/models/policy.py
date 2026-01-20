import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Attention-based Policy Model
# ----------------------------
class AttentionPolicy(nn.Module):
    def __init__(self, node_dim=2, embed_dim=128):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, nodes, current_node_idx, visited_mask):
        """
        nodes: [N, node_dim]             node features (e.g. coordinates)
        current_node_idx: int            current position
        visited_mask: [N] (bool)         True = already visited
        """        
        
        # Embed nodes
        h = self.node_embed(nodes)
        
        # Query = embedding of current node
        q = self.query(h[current_node_idx])
        
        # Keys = all nodes
        k = self.key(h)
        
         # Attention scores
        scores = torch.matmul(k, q)
        
        # Masking: visited nodes get -inf score
        scores = scores.masked_fill(visited_mask, float("-inf"))
        
        # probability of choosing next node
        probs = F.softmax(scores, dim=0)
        return probs
    
    
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

    def forward(self, nodes, current_idx, visited_mask):
        """
        nodes: [N, node_dim]
        current_idx: int
        visited_mask: [N] bool
        """

        # ======================
        # 1️ Encode graph nodes
        # ======================
        h = self.node_embed(nodes)        # [N, D]

        # ======================
        # 2️ Graph message passing (mean aggregation)
        # ======================
        graph_context = h.mean(dim=0, keepdim=True)   # [1, D]
        h = h + self.msg_linear(graph_context)        # [N, D]

        # ======================
        # 3️ Pointer attention
        # ======================
        q = self.query(h[current_idx])    # [D]
        k = self.key(h)                   # [N, D]

        scores = torch.matmul(k, q)       # [N]
        scores = scores.masked_fill(visited_mask, -1e9)

        probs = F.softmax(scores, dim=0)
        return probs    