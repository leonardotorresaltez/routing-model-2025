"""
Graph-aware policy for fleet routing using Graph Pointer Network.

Integrates Graph Pointer Network with SB3 by converting flat observations
to graph representations and processing them through the graph architecture.
Leverages SB3's built-in MultiDiscrete handling for action spaces.
"""

from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from model.tsp_agent.graph_pointer_network_model import GraphPointerNetwork
from model.graph_converter import observation_to_graph


class GraphAwareFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using Graph Pointer Network for fleet routing.
    
    Processes flat observations by:
    1. Converting to graph representation (node features + adjacency)
    2. Running through GraphPointerNetwork to get node attention scores
    3. Combining graph outputs with MLP features for final embeddings
    
    This enables explicit spatial reasoning about customer and truck locations.
    """

    def __init__(
        self,
        observation_space,
        num_trucks: int = 4,
        num_customers: int = 15,
        hidden_dim: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU,
        k_neighbors: int = 5,
    ):
        super().__init__(observation_space, features_dim=hidden_dim)
        
        self.num_trucks = num_trucks
        self.num_customers = num_customers
        self.hidden_dim = hidden_dim
        self.obs_size = observation_space.shape[0]
        self.k_neighbors = k_neighbors
        
        self.graph_pointer_network = GraphPointerNetwork(
            in_dim=2,
            hidden_dim=hidden_dim
        )
        
        self.graph_to_features = nn.Sequential(
            nn.Linear(self.num_trucks + self.num_customers, 128),
            activation_fn(),
            nn.Linear(128, hidden_dim),
            activation_fn(),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.obs_size, 128),
            activation_fn(),
            nn.Linear(128, hidden_dim),
            activation_fn(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation_fn(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features by combining graph and MLP representations.
        
        Pipeline:
        1. Convert flat observation to graph (node features + adjacency)
        2. Run through GraphPointerNetwork to get attention scores over nodes
        3. Use scores as graph-aware features
        4. Combine with MLP features via fusion network
        
        Args:
            observations: Batch of observations, shape [batch_size, obs_dim]
        
        Returns:
            Graph-aware feature embeddings of shape [batch_size, hidden_dim]
        """
        batch_size = observations.shape[0]
        
        graph_features_list = []
        
        for i in range(batch_size):
            obs_np = observations[i].detach().cpu().numpy()
            
            node_features, adjacency_matrix = observation_to_graph(
                obs_np,
                self.num_trucks,
                self.num_customers,
                k_neighbors=self.k_neighbors
            )
            
            node_features = node_features.to(observations.device)
            adjacency_matrix = adjacency_matrix.to(observations.device)
            
            graph_attention = self.graph_pointer_network(node_features, adjacency_matrix)
            
            graph_features_list.append(graph_attention)
        
        graph_features_batch = torch.stack(graph_features_list, dim=0)
        
        graph_embeddings = self.graph_to_features(graph_features_batch)
        
        mlp_embeddings = self.mlp(observations)
        
        fused_features = torch.cat([graph_embeddings, mlp_embeddings], dim=-1)
        combined_features = self.fusion(fused_features)
        
        return combined_features


class GraphPointerPolicy(ActorCriticPolicy):
    """
    Policy for fleet routing using Graph Pointer Network architecture.
    
    Integrates GraphPointerNetwork into SB3's PPO by:
    1. Converting flat observations to graph representations
    2. Processing through GraphPointerNetwork
    3. Combining graph-aware features with MLP features
    4. Using SB3's built-in MultiDiscrete action handling
    
    This enables explicit spatial reasoning during policy learning.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch: Optional[Dict[str, list]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        num_trucks: int = 4,
        num_customers: int = 15,
        hidden_dim: int = 64,
        k_neighbors: int = 5,
        **kwargs
    ):
        """
        Initialize GraphPointerPolicy.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space (MultiDiscrete)
            lr_schedule: Learning rate schedule
            net_arch: Network architecture for policy/value heads
            activation_fn: Activation function class
            num_trucks: Number of trucks in the routing problem
            num_customers: Number of customers to deliver to
            hidden_dim: Hidden dimension for embeddings
            k_neighbors: Number of nearest neighbors for graph adjacency
            **kwargs: Additional arguments for ActorCriticPolicy
        """
        if net_arch is None:
            net_arch = dict(pi=[hidden_dim], vf=[hidden_dim])
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=GraphAwareFeaturesExtractor,
            features_extractor_kwargs=dict(
                num_trucks=num_trucks,
                num_customers=num_customers,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                k_neighbors=k_neighbors
            ),
            **kwargs
        )
