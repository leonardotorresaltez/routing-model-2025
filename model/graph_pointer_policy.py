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
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from model.graph_converter import observation_to_graph
from model.tsp_agent.graph_pointer_network_model import GraphPointerNetwork

# class GraphAwareFeaturesExtractor(BaseFeaturesExtractor):
#     """
#     Feature extractor using Graph Pointer Network for fleet routing.
    
#     Processes flat observations by:
#     1. Converting to graph representation (node features + adjacency)
#     2. Running through GraphPointerNetwork to get node attention scores
#     3. Combining graph outputs with MLP features for final embeddings
    
#     This enables explicit spatial reasoning about customer and truck locations.
#     """
    
    
    
#     # In one sentence: A smart decision-making system that helps trucks decide which customers to deliver to, by visualizing the problem as a graph of connected locations.

#     # The Real-World Problem
#     # Imagine you have:

#     # 4 delivery trucks at different locations
#     # 15 customers scattered across a city
#     # Each truck has a volume limit (capacity)
#     # Each customer needs a package delivery
#     # The question: Which customers should each truck deliver to minimize total distance traveled?

#     # How the Graph Pointer Network Works
#     # Step 1: See the Problem as a Map (Graph)

#     # Instead of treating the problem as just numbers, the system converts it into a spatial graph
#     # Nodes = trucks + customers (19 points on a map)
#     # Connections = nearby locations linked together (if they're close in distance)
#     # Step 2: Use Attention to Find Important Customers

#     # The system uses "attention" (like human eyes focusing on relevant things)
#     # It looks at each truck's position and asks: "Which unvisited customers are most important for me to visit next?"
#     # It scores all customers based on their location and feasibility
#     # Step 3: Combine with Traditional Learning

#     # The graph insights are combined with traditional neural network features
#     # Both approaches vote together to make the best decision
#     # Step 4: Output the Decision

#     # Each truck gets a decision: "Go to customer 3" or "Return to depot"


#     def __init__(
#         self,
#         observation_space,
#         num_trucks: int = 4,
#         num_customers: int = 15,
#         hidden_dim: int = 64,
#         activation_fn: Type[nn.Module] = nn.ReLU, # The graph pointer network is the default feature extractor, but net_arch controls what policy/value networks are built on top of those extracted features.
#         k_neighbors: int = 5, # ASKJORGE # FIXME
#     ):
#         super().__init__(observation_space, features_dim=hidden_dim)
        
#         self.num_trucks = num_trucks
#         self.num_customers = num_customers
#         self.hidden_dim = hidden_dim
#         self.obs_size = observation_space.shape[0]
#         self.k_neighbors = k_neighbors
        
#         self.graph_pointer_network = GraphPointerNetwork(
#             in_dim=2,
#             hidden_dim=hidden_dim
#         )
        
#         self.graph_to_features = nn.Sequential(
#             nn.Linear(self.num_trucks + self.num_customers, 128),
#             activation_fn(),
#             nn.Linear(128, hidden_dim),
#             activation_fn(),
#         )
        
#         self.mlp = nn.Sequential(
#             nn.Linear(self.obs_size, 128),
#             activation_fn(),
#             nn.Linear(128, hidden_dim),
#             activation_fn(),
#         )
        
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             activation_fn(),
#         )

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         """
#         Extract features by combining graph and MLP representations.
        
#         Pipeline:
#         1. Convert flat observation to graph (node features + adjacency)
#         2. Run through GraphPointerNetwork to get attention scores over nodes
#         3. Use scores as graph-aware features
#         4. Combine with MLP features via fusion network
        
#         Args:
#             observations: Batch of observations, shape [batch_size, obs_dim]
        
#         Returns:
#             Graph-aware feature embeddings of shape [batch_size, hidden_dim]
#         """
#         batch_size = observations.shape[0]
        
#         graph_features_list = []
        
#         for i in range(batch_size):
#             obs_np = observations[i].detach().cpu().numpy()
            
#             node_features, adjacency_matrix = observation_to_graph(
#                 obs_np,
#                 self.num_trucks,
#                 self.num_customers,
#                 k_neighbors=self.k_neighbors
#             )
            
#             node_features = node_features.to(observations.device)
#             adjacency_matrix = adjacency_matrix.to(observations.device)
            
#             graph_attention = self.graph_pointer_network(node_features, adjacency_matrix)
            
#             graph_features_list.append(graph_attention)
        
#         graph_features_batch = torch.stack(graph_features_list, dim=0)
        
#         graph_embeddings = self.graph_to_features(graph_features_batch)
        
#         mlp_embeddings = self.mlp(observations)
        
#         fused_features = torch.cat([graph_embeddings, mlp_embeddings], dim=-1)
#         combined_features = self.fusion(fused_features)
        
#         return combined_features


class SimpleGraphFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simplified graph-only feature extractor for fleet routing.
    
    Uses only GraphPointerNetwork for feature extraction without additional
    MLP branches or fusion layers. Suitable for keeping the model simple while
    leveraging graph structure of the routing problem.
    """
    
    def __init__(
        self,
        observation_space,
        num_trucks: int = 4,
        num_customers: int = 15,
        hidden_dim: int = 64,
        k_neighbors: int = None,
    ):
        super().__init__(observation_space, features_dim=hidden_dim)
        
        self.num_trucks = num_trucks
        self.num_customers = num_customers
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors if k_neighbors is not None else num_trucks + num_customers
        
        self.graph_pointer_network = GraphPointerNetwork(
            in_dim=2,
            hidden_dim=hidden_dim
        )
        
        self.projection = nn.Linear(num_trucks + num_customers, hidden_dim)
        
        # Feature Extractor Role, which:

        # Extracts/computes feature representations from observations
        # Should output meaningful embeddings (shape: [batch_size, hidden_dim])
        # These will feed into policy and value heads

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features using only GraphPointerNetwork.
        
        Args:
            observations: Batch of observations, shape [batch_size, obs_dim]
        
        Returns:
            Graph-based feature embeddings of shape [batch_size, hidden_dim]
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
        features = self.projection(graph_features_batch)
        
        return features


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
        k_neighbors: int = None,
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
            k_neighbors: Number of nearest neighbors for graph adjacency (default: None for full connectivity)
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


class SimpleGraphPointerPolicy(ActorCriticPolicy):
    """
    Simplified policy using graph-only feature extraction with action masking.
    
    Uses SimpleGraphFeaturesExtractor for cleaner, graph-focused learning
    without MLP fusion branches. Ideal for keeping the model simple while
    maintaining graph-based spatial reasoning.
    
    Applies feasibility masking inside the policy network (during distribution
    creation) like the notebook approach, ensuring only valid actions get
    non-zero probability.
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
        k_neighbors: int = None,
        **kwargs
    ):
        """
        Initialize SimpleGraphPointerPolicy.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space (MultiDiscrete)
            lr_schedule: Learning rate schedule
            net_arch: Network architecture for policy/value heads
            activation_fn: Activation function class
            num_trucks: Number of trucks in the routing problem
            num_customers: Number of customers to deliver to
            hidden_dim: Hidden dimension for embeddings
            k_neighbors: Number of nearest neighbors for graph adjacency (default: None for full connectivity)
            **kwargs: Additional arguments for ActorCriticPolicy
        """
        self.num_trucks = num_trucks
        self.num_customers = num_customers
        self._last_observations = None
        
        if net_arch is None:
            net_arch = dict(pi=[hidden_dim], vf=[hidden_dim])
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=SimpleGraphFeaturesExtractor,
            features_extractor_kwargs=dict(
                num_trucks=num_trucks,
                num_customers=num_customers,
                hidden_dim=hidden_dim,
                k_neighbors=k_neighbors
            ),
            **kwargs
        )
    
    def _extract_feasibility_mask(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract feasibility_mask from flattened observations.
        
        Observations layout:
        - Positions: 2*T
        - Loads: T
        - Capacities: T
        - Utilization: T
        - Customer locations: 2*N
        - Customer volumes: N
        - Unvisited mask: N
        - Feasibility mask: T*(N+1) where N+1 includes depot option
        
        Total before mask: 5*T + 4*N
        
        Args:
            observations: Batch of flattened observations
            
        Returns:
            Feasibility mask tensor of shape [batch_size, T, N+1]
        """
        T = self.num_trucks
        N = self.num_customers
        
        obs_size = 5 * T + 4 * N
        mask_start = obs_size
        
        mask_flat = observations[:, mask_start:]
        mask = mask_flat.reshape(-1, T, N + 1)
        
        return mask
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass that stores observations for masking and calls parent implementation.
        
        Args:
            obs: Observations tensor
            deterministic: Whether to use deterministic action selection
            
        Returns:
            actions, values
        """
        self._last_observations = obs
        return super().forward(obs, deterministic=deterministic)
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Get action distribution from policy latent with feasibility masking.
        
        Applies -infinity masking to logits for infeasible actions before
        creating distribution, ensuring only valid actions get sampled.
        
        Args:
            latent_pi: Policy latent features [batch_size, hidden_dim]
            
        Returns:
            Masked action distribution (MultiCategorical)
        """
        mean_actions = self.action_net(latent_pi)
        
        if self._last_observations is not None:
            try:
                feasibility_mask = self._extract_feasibility_mask(self._last_observations)
                feasibility_mask = feasibility_mask.bool()
                
                mean_actions_masked = mean_actions.clone()
                batch_size = mean_actions.shape[0]
                
                for b in range(batch_size):
                    for t in range(self.num_trucks):
                        invalid_actions = ~feasibility_mask[b, t]
                        mean_actions_masked[b, t][invalid_actions] = float('-inf')
                
                mean_actions = mean_actions_masked
            except Exception as e:
                pass
        
        return self.action_dist.proba_distribution(mean_actions)
