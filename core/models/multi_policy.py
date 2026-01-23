import torch
import torch.nn as nn
import torch.nn.functional as F

class PlannerPolicy(nn.Module):
    def __init__(self, node_dim=3, embed_dim=128, num_trucks=3):
        super().__init__()
        self.num_trucks = num_trucks

        self.node_embed = nn.Linear(node_dim, embed_dim)
        self.global_query = nn.Parameter(torch.randn(embed_dim))
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, node_features, edge_index, edge_attr):
        h = self.node_embed(node_features)          # (N, D)
        attn_scores = torch.matmul(h, self.global_query)   # (N,)
        node_scores = self.out_proj(h).squeeze(-1)         # (N,)
        logits = attn_scores + node_scores                 # (N,)
        planner_logits = logits.unsqueeze(0).repeat(self.num_trucks, 1)
        return F.softmax(planner_logits, dim=1)


class TruckPolicy(nn.Module):
    def __init__(self, node_dim=3, embed_dim=128):
        super().__init__()

        self.node_embed = nn.Linear(node_dim, embed_dim)
        self.truck_pos_embed = nn.Embedding(256, embed_dim)
        self.suggest_embed = nn.Embedding(256, embed_dim)

        self.fc1 = nn.Linear(embed_dim * 3, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)

    def forward(self, node_features, edge_index, edge_attr, truck_pos, planner_suggestion):
        h = self.node_embed(node_features)

        truck_emb = self.truck_pos_embed(truck_pos)
        suggest_emb = self.suggest_embed(planner_suggestion)

        truck_rep = truck_emb.unsqueeze(0).repeat(h.size(0), 1)
        suggest_rep = suggest_emb.unsqueeze(0).repeat(h.size(0), 1)

        combined = torch.cat([h, truck_rep, suggest_rep], dim=-1)
        x = F.relu(self.fc1(combined))
        scores = self.fc2(x).squeeze(-1)

        return F.softmax(scores, dim=0)