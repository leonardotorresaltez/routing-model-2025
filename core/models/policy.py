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
        
        #embed nodes , shape is N=number of nodes, D=embed_dim
        h = self.node_embed(nodes)        # [N, D]
        # example nodes tensor([[0.1, 0.2],  # Nodo 1
        #                       [0.3, 0.4],  # Nodo 2
        #                       [0.5, 0.6],  # Nodo 3
        #                       [0.7, 0.8],  # Nodo 4
        #                       [0.9, 1.0]]) # Nodo 5
        
        #example h embedding tensor after node_embed layer
        #torch.tensor([[ 0.1234, -0.5678, ..., 0.9876],  # Embedding del nodo 1 [128 valores]
        #[ 0.2345, -0.6789, ..., 1.0987],  # Embedding del nodo 2
        #[ 0.3456, -0.7890, ..., 1.2098],  # Embedding del nodo 3
        #[ 0.4567, -0.8901, ..., 1.3209],  # Embedding del nodo 4
        #[ 0.5678, -0.9012, ..., 1.4320]]) # Embedding del nodo 5        
        
        # ======================
        # 2️ Graph message passing (mean aggregation)
        # ======================
        
        #mean of embedded nodes, dim=0 means in which dimension to take the mean
        #example:  tensor([[ 0.3456, -0.7674, ..., 1.2098]])   - one vector of size D
        graph_context = h.mean(dim=0, keepdim=True)   # [1, D]
        #update node embeddings with graph context
        h = h + self.msg_linear(graph_context)        # [N, D]



        # ======================
        # 3️ Pointer attention
        # ======================
        
        #query based on current node
        q = self.query(h[current_idx])    # [D]
        #keys for all nodes
        k = self.key(h)                   # [N, D]

        #compute attention scores
        scores = torch.matmul(k, q)       # [N]
        scores = scores.masked_fill(visited_mask, -1e9)

        probs = F.softmax(scores, dim=0)
        return probs    