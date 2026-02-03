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
    
# ----------------------------
# TODO does not perform better than GraphPointer
# MultiHeadGraphPointer Policy Model
# ----------------------------       

class MultiHeadGraphPointerPolicy(nn.Module):
    def __init__(self, node_dim=2, embed_dim=128):
        super().__init__()

        # ---- Shared node embedding ----
        self.node_embed = nn.Linear(node_dim, embed_dim)

        # ---- LOCAL HEAD (distance-like) ----
        self.local_query = nn.Linear(embed_dim, embed_dim)
        self.local_key = nn.Linear(embed_dim, embed_dim)

        # ---- CLUSTER HEAD (global structure) ----
        self.cluster_q = nn.Linear(embed_dim, embed_dim)
        self.cluster_k = nn.Linear(embed_dim, embed_dim)
        self.cluster_v = nn.Linear(embed_dim, embed_dim)

        # ---- Fusión de heads ----
        # concatenamos scores de ambas cabezas y proyectamos a un único valor por nodo
        self.out_proj = nn.Linear(2, 1)

    def forward(self, nodes, current_node, visited_mask):
        """
        nodes:        [N, 2]
        current_node: int
        visited_mask: [N] bool
        """
        visited_mask = visited_mask.bool()
        h = self.node_embed(nodes)  # [N, D]

        # ==============================
        # HEAD 1 — LOCAL POINTER
        # ==============================
        q_local = self.local_query(h[current_node])  # [D]
        k_local = self.local_key(h)                  # [N, D]
        scores_local = torch.matmul(k_local, q_local)  # [N]

        # ==============================
        # HEAD 2 — CLUSTER / GLOBAL SELF-ATTENTION
        # ==============================
        Q = self.cluster_q(h)  # [N, D]
        K = self.cluster_k(h)  # [N, D]
        V = self.cluster_v(h)  # [N, D]

        attn_scores = torch.matmul(Q, K.T) / (h.size(1) ** 0.5)  # [N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        cluster_context = attn_weights @ V  # [N, D]

        # Global score por nodo
        scores_cluster = torch.sum(cluster_context * h, dim=1)  # [N]

        # ==============================
        # CONCAT + PROYECCIÓN
        # ==============================
        # Concatenamos los scores de las dos cabezas por nodo
        combined = torch.stack([scores_local, scores_cluster], dim=1)  # [N, 2]
        scores = self.out_proj(combined).squeeze(-1)  # [N]

        # ==============================
        # MASK VISITED NODES
        # ==============================
        scores = scores.masked_fill(visited_mask, -1e9)
        probs = F.softmax(scores, dim=0)
        return probs  