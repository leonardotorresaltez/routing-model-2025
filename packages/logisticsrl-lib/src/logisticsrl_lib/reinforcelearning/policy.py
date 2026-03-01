import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# GraphPointer Policy Model (Actor-Critic)
# ----------------------------    
class GraphPointerPolicy(nn.Module):
    def __init__(self, cfg, node_dim, embed_dim):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim  # Save for the attention scaling factor
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim) # Deep embedding
        )

        # -------------------------------------------------
        # NO-OP Learnable Parameters
        # -------------------------------------------------
        self.noop_key = nn.Parameter(torch.randn(embed_dim) / math.sqrt(embed_dim))
        # self.noop_bias = nn.Parameter(torch.tensor([-50.0])) 
        self.noop_bias = nn.Parameter(torch.tensor([0.0])) 

        # Attention Context
        self.ctx_query = nn.Linear(embed_dim, embed_dim)
        self.ctx_key = nn.Linear(embed_dim, embed_dim)
        self.ctx_value = nn.Linear(embed_dim, embed_dim)

        # Pointer mechanism (Actor)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        
        # -------------------------------------------------
        # NEW: Critic Head (Value Network)
        # -------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, nodes: torch.Tensor , current_node: int, visited_mask: torch.Tensor):
        visited_mask = visited_mask.bool()
        # num_nodes = nodes.shape[0]
        # active_trucks = nodes[:, 2].sum()
        # if active_trucks < 5: raise(Exception(f"This is good! The noop mechanism is working. Active trucks: {active_trucks}"))
        # truck_positions = nodes[:, 2:].T
        # print(f"DEBUG: Truck positions in forward pass: {truck_positions.cpu().numpy()}")
        # print(f"DEBUG: Visited mask sum: {visited_mask.sum().item()}")
        # print(f"DEBUG: Visited mask sum inputs: {nodes[:, 3].sum().item()}")
        # print(f"DEBUG: Current trucks sum: {nodes[:, 2].sum().item()}")
        # breakpoint()


        # 1. Embed the nodes
        h = self.node_embed(nodes)           # [N, D]
        
        # 2. Compute Dynamic Graph Context (Attention)
        curr_q = self.ctx_query(h[current_node])  # [D]
        
        ctx_k = self.ctx_key(h)                   # [N, D]
        ctx_v = self.ctx_value(h)                 # [N, D]
        
        ctx_scores = torch.matmul(ctx_k, curr_q) / math.sqrt(self.embed_dim)  # [N]
        ctx_scores = ctx_scores.masked_fill(visited_mask, -1e9)
        
        ctx_weights = F.softmax(ctx_scores, dim=0)    # [N]
        graph_ctx = torch.matmul(ctx_weights, ctx_v)  # [D]

        # -------------------------------------------------
        # NEW: Critic Value Prediction
        # -------------------------------------------------
        # We evaluate the state based on the current node and the graph context
        state_representation = h[current_node] + graph_ctx
        state_value = self.value_head(state_representation).squeeze() # Scalar

        # 3. Final Pointer Mechanism (Actor)
        query = self.query(state_representation)         # [D]
        keys = self.key(h)                               # [N, D]

        # FIXED: Add scaling factor here
        scores = torch.matmul(keys, query) / math.sqrt(self.embed_dim) 
        scores = scores.masked_fill(visited_mask, -1e9)

        # 4. Add the NO-OP Score
        # FIXED: Add scaling factor here too
        noop_score = (torch.dot(query, self.noop_key) / math.sqrt(self.embed_dim)) + self.noop_bias
        noop_score = noop_score.view(1)
        if getattr(self.cfg, 'debug', False): print(f"DEBUG: Action scores before masking: {scores.cpu().detach().numpy()}")
        # if getattr(self.cfg, 'debug', False): print(f"DEBUG: Action scores before masking:")
        if getattr(self.cfg, 'debug', False) or visited_mask.sum().item() == 0: 
            print(f"DEBUG: NO-OP score: {noop_score.item()}")
        
        final_scores = torch.cat([scores, noop_score], dim=0)          # [N + 1]
        probs = F.softmax(final_scores, dim=0)                         # [N + 1]
        
        # if getattr(self.cfg, 'debug', False): 
            # print(f"DEBUG: Action probabilities shape: {probs.cpu().detach().numpy().shape} | Sum: {probs.sum().item():.4f}")
            # print(f"DEBUG: Action probabilities: {probs.cpu().detach().numpy()}")
            
        return probs, state_value # <--- NEW: Return both Actor and Critic outputs