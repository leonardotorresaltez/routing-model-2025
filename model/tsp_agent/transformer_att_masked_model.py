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
        h = self.node_embed(nodes)       # [N, embed_dim]

        # Query = embedding of current node
        q = self.query(h[current_node_idx])   # [embed_dim]

        # Keys = all nodes
        k = self.key(h)                       # [N, embed_dim]

        # Attention scores
        scores = torch.matmul(k, q)           # [N]

        # Mask visited nodes, the agent cannot revisit them
        scores = scores.masked_fill(visited_mask, float("-inf"))

        # Policy = probability of choosing next node
        # política estocástica
        probs = F.softmax(scores, dim=0)

        return probs
    
    

#N = 5
#node_dim = 2      # e.g. (x, y)
#embed_dim = 32

# random coordinates for nodes
#nodes = torch.rand(N, node_dim)  
#print("Input nodes:", nodes)

#current_node = 0

#visited_mask = torch.tensor([True, False, False, False, False])

#policy = AttentionPolicy(node_dim, embed_dim)

#probs = policy(nodes, current_node, visited_mask)

#action = torch.multinomial(probs, 1).item()

#print("Action probabilities:", probs)
#print("Selected next node:", action)
