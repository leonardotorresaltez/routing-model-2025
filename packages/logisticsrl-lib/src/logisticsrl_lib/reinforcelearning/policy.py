import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




# ----------------------------
# FactorizedFleetPolicy Policy Model
# ----------------------------    
class FactorizedFleetPolicy(nn.Module):
    #_SUM_OTHER_DIM = 2
    
    def __init__(self, cfg, embed_dim=128,input_features_size=10,knn_k=10):
        super().__init__()
        self.cfg = cfg

        self.embed_dim = embed_dim
        self.knn_k =knn_k
       
        # Feature embedding — initialized at construction so the optimizer tracks it
        self.features = nn.Linear(input_features_size, embed_dim)
        # self.features = None
        # Simple graph message passing (1 step)
        self.msg_linear = nn.Linear(embed_dim, embed_dim)

        # node selector pointer mechanism
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        
        # truck selector pointer mechanism
        self.truck_query = nn.Linear(embed_dim, embed_dim)
        self.truck_key = nn.Linear(embed_dim, embed_dim)

        # critic: value head V(s)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )        
    # ----------------------------
    # KNN Graph Builder
    # ----------------------------    
    def build_edge_index(self,coords):
       device = coords.device
       k = self.knn_k
        
       num_nodes = coords.shape[0]
        
        #Pairwise distance
       dist = torch.cdist(coords,coords)
        
        #K nearest neighbours(skipping itself)
       knn =dist.topk(k+1,largest=False).indices[:,1:]
        
        #Directed edges source->destination
       src = torch.arange(num_nodes,device=device).unsqueeze(1).repeat(1,k).reshape(-1)
       dst = knn.reshape(-1)
        
       edge_index = torch.stack([src,dst],dim =0)
        
        #Now reverse edges destination->src
       edge_index_rev = edge_index.flip(0)
       edge_index = torch.cat([edge_index,edge_index_rev],dim=1)
        
        #Add self loops
       self_loops = torch.arange(num_nodes,device=device)
       self_loops = torch.stack([self_loops,self_loops],dim=0)
       edge_index = torch.cat([edge_index,self_loops],dim=1)
        
       return edge_index    # ----------------------------
    # Graph message passing
    # ----------------------------
    def graph_message_passing(self,h,edge_index):
        src,dst=edge_index
        messages = h[src]
        
        #sum messages into each node
        agg = torch.zeros_like(h)
        agg.index_add_(0,dst,messages)
        
        # normalize by degree (VERY IMPORTANT)
        deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).unsqueeze(1)
        agg = agg / deg

        out =F.relu(self.msg_linear(agg))
        # apply linear layer + activation
        return h+out


         
    def forward(self, observation_space_as_features: torch.Tensor, truck_positions: np.array, visited_mask: torch.Tensor, inactive_trucks_mask: torch.Tensor,coords):
        """
        observation_space_as_features: [N, size_dim]
        truck_positions: np.array
        visited_mask: [T, N] bool  (True = forbidden)
        inactive_trucks_mask: [T] bool (True = inactive)
        """
        
        # Dynamically initialize self.features if it is None
        #if self.features is None:
        #    input_features_size = observation_space_as_features.size(1)  # number of nodes dynamically
        #    self.features = nn.Linear(input_features_size, self.embed_dim)        

        h = self.features(observation_space_as_features)           # [N, D]
        #Build KNN and do message passing 
        edge_index = self.build_edge_index(coords)
        h = self.graph_message_passing(h,edge_index)
        
        #Global graph now
        graph_ctx = h.mean(0)  # [D]
        
        #Truck Embeddings
        if isinstance(truck_positions,torch.Tensor):
            truck_pos_tensor = truck_positions.to(h.device)
        else:
            truck_pos_tensor = torch.tensor(truck_positions,device=h.device, dtype=torch.long)
        truck_h = h[truck_pos_tensor] 

        visited_mask = visited_mask.bool()
       

        # ---------- TRUCK SELECTION ----------
        tq = self.truck_query(truck_h + graph_ctx) # [T,D]
        tk = self.truck_key(truck_h)               # [T,D]

        truck_scores = torch.matmul(tq, tk.T).diag()  # [T]
        truck_scores = truck_scores.masked_fill(inactive_trucks_mask, -1e9)  # Apply truck time mask
        truck_probs = F.softmax(truck_scores, dim=0)  
        
        # ---------- NODE SELECTION ----------
        query = self.query(truck_h + graph_ctx)  # [D]
        keys = self.key(h)                               # [N, D]

        node_scores = torch.matmul(query, keys.T)               # [N]       
        node_scores = node_scores.masked_fill(visited_mask, -1e9)    
        node_probs = F.softmax(node_scores, dim=-1)  # Corrected dimension for softmax to apply along the last axis     
 
        
        # critic: V(s) from graph context
        value = self.value_head(graph_ctx).squeeze(-1)  # scalar

        if self.cfg.debug: print(f"DEBUG: Action probabilities shape: {node_probs.cpu().detach().numpy().shape} | Sum: {node_probs.sum().item():.4f}")
        return truck_probs, node_probs, value