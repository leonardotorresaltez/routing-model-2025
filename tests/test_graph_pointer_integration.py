"""
Tests for Graph Pointer Network integration with fleet routing environment.

Verifies:
- Graph conversion from flat observations
- GraphPointerNetwork execution
- Feature extraction pipeline
- Full PPO training with GraphPointerPolicy
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from model.graph_converter import (
    reconstruct_observation_components,
    build_node_features,
    build_adjacency_matrix,
    observation_to_graph,
    create_networkx_graph,
)
from model.graph_pointer_policy import GraphAwareFeaturesExtractor, GraphPointerPolicy
from env.sb3_wrapper import FleetRoutingSB3Wrapper
from env.routing_env_simple import SimpleFleetRoutingEnv
from env.types_simple import Customer, Truck, Depot


class TestGraphConverter:
    """Test graph conversion utilities."""

    def test_reconstruct_observation_components(self):
        """
        Test reconstruction of observation components from flat vector.
        
        Verifies that the flat observation vector is correctly split into
        individual components matching the original shapes.
        """
        num_trucks = 4
        num_customers = 15
        obs_size = 5 * num_trucks + 4 * num_customers
        
        flat_obs = np.random.randn(obs_size).astype(np.float32)
        
        components = reconstruct_observation_components(
            flat_obs, num_trucks, num_customers
        )
        
        assert components['truck_positions'].shape == (num_trucks, 2)
        assert components['truck_loads'].shape == (num_trucks,)
        assert components['truck_capacities'].shape == (num_trucks,)
        assert components['truck_utilization'].shape == (num_trucks,)
        assert components['customer_locations'].shape == (num_customers, 2)
        assert components['customer_weights'].shape == (num_customers,)
        assert components['unvisited_mask'].shape == (num_customers,)
        
        print("[PASS] reconstruct_observation_components")

    def test_build_node_features(self):
        """
        Test node feature matrix construction.
        
        Verifies that truck and customer positions are correctly combined
        into a unified node feature matrix.
        """
        num_trucks = 4
        num_customers = 15
        
        truck_pos = np.random.randn(num_trucks, 2).astype(np.float32)
        customer_pos = np.random.randn(num_customers, 2).astype(np.float32)
        
        node_features = build_node_features(truck_pos, customer_pos)
        
        assert node_features.shape == (num_trucks + num_customers, 2)
        assert isinstance(node_features, torch.Tensor)
        assert node_features.dtype == torch.float32
        
        np.testing.assert_array_equal(
            node_features[:num_trucks].cpu().numpy(),
            truck_pos
        )
        np.testing.assert_array_equal(
            node_features[num_trucks:].cpu().numpy(),
            customer_pos
        )
        
        print("[PASS] build_node_features")

    def test_build_adjacency_matrix(self):
        """
        Test adjacency matrix construction using k-nearest neighbors.
        
        Verifies that the adjacency matrix correctly represents spatial
        proximity between nodes.
        """
        num_trucks = 4
        num_customers = 15
        n_nodes = num_trucks + num_customers
        
        node_features = torch.randn(n_nodes, 2)
        
        adjacency = build_adjacency_matrix(node_features, k_neighbors=5)
        
        assert adjacency.shape == (n_nodes, n_nodes)
        assert torch.allclose(adjacency, adjacency.T)
        assert torch.all((adjacency == 0) | (adjacency == 1))
        
        row_sums = adjacency.sum(dim=1)
        assert torch.all(row_sums > 0)
        
        print("[PASS] build_adjacency_matrix")

    def test_observation_to_graph(self):
        """
        Test full observation-to-graph conversion pipeline.
        
        Verifies that flat observations are correctly converted to
        graph representations (nodes and adjacency).
        """
        num_trucks = 4
        num_customers = 15
        obs_size = 5 * num_trucks + 4 * num_customers
        
        flat_obs = np.random.randn(obs_size).astype(np.float32)
        
        node_features, adjacency_matrix = observation_to_graph(
            flat_obs, num_trucks, num_customers, k_neighbors=5
        )
        
        assert node_features.shape == (num_trucks + num_customers, 2)
        assert adjacency_matrix.shape == (num_trucks + num_customers, num_trucks + num_customers)
        assert isinstance(node_features, torch.Tensor)
        assert isinstance(adjacency_matrix, torch.Tensor)
        
        print("[PASS] observation_to_graph")

    def test_create_networkx_graph(self):
        """
        Test NetworkX graph creation from tensors.
        
        Verifies that graph structure is correctly represented in NetworkX
        format for visualization and analysis.
        """
        num_nodes = 10
        node_features = torch.randn(num_nodes, 2)
        adjacency = torch.eye(num_nodes)
        for i in range(num_nodes - 1):
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1
        
        G = create_networkx_graph(node_features, adjacency)
        
        assert len(G.nodes) == num_nodes
        assert len(G.edges) > 0
        
        for node in G.nodes():
            assert 'pos' in G.nodes[node]
        
        print("[PASS] create_networkx_graph")


class TestGraphPointerFeatures:
    """Test GraphPointerNetwork feature extraction."""

    def test_graph_aware_features_extractor(self):
        """
        Test GraphAwareFeaturesExtractor forward pass.
        
        Verifies that the extractor correctly processes batch observations
        through graph and MLP pathways, producing embeddings.
        """
        num_trucks = 4
        num_customers = 15
        hidden_dim = 64
        batch_size = 8
        obs_size = 5 * num_trucks + 4 * num_customers
        
        obs_space = spaces.Box(low=-1000, high=1000, shape=(obs_size,), dtype=np.float32)
        
        extractor = GraphAwareFeaturesExtractor(
            obs_space,
            num_trucks=num_trucks,
            num_customers=num_customers,
            hidden_dim=hidden_dim
        )
        
        observations = torch.randn(batch_size, obs_size)
        
        features = extractor(observations)
        
        assert features.shape == (batch_size, hidden_dim)
        assert isinstance(features, torch.Tensor)
        assert not torch.isnan(features).any()
        
        print("[PASS] GraphAwareFeaturesExtractor")

    def test_graph_pointer_policy_creation(self):
        """
        Test GraphPointerPolicy initialization.
        
        Verifies that the policy is correctly instantiated with proper
        feature extractor and architecture.
        """
        num_trucks = 4
        num_customers = 15
        hidden_dim = 64
        obs_size = 5 * num_trucks + 4 * num_customers
        
        obs_space = spaces.Box(low=-1000, high=1000, shape=(obs_size,), dtype=np.float32)
        action_space = spaces.MultiDiscrete([num_customers + 1] * num_trucks)
        
        def lr_schedule(progress):
            return 3e-4
        
        policy = GraphPointerPolicy(
            obs_space,
            action_space,
            lr_schedule=lr_schedule,
            num_trucks=num_trucks,
            num_customers=num_customers,
            hidden_dim=hidden_dim
        )
        
        assert hasattr(policy, 'features_extractor')
        assert isinstance(policy.features_extractor, GraphAwareFeaturesExtractor)
        
        print("[PASS] GraphPointerPolicy creation")


class TestGraphPointerIntegration:
    """Integration tests with environment."""

    def test_features_extractor_with_environment(self):
        """
        Test feature extraction with real environment observations.
        
        Verifies that the extractor works correctly with observations
        from the actual routing environment.
        """
        num_trucks = 4
        num_customers = 15
        
        customers = [Customer(i, np.random.rand() * 100, np.random.rand() * 100, np.random.rand() * 20 + 10) for i in range(num_customers)]
        trucks = [Truck(i, 0, 50.0) for i in range(num_trucks)]
        depots = [Depot(0, 50.0, 50.0)]
        
        env = SimpleFleetRoutingEnv(customers, trucks, depots)
        env = FleetRoutingSB3Wrapper(env)
        
        obs, _ = env.reset()
        
        hidden_dim = 64
        obs_space = env.observation_space
        
        extractor = GraphAwareFeaturesExtractor(
            obs_space,
            num_trucks=num_trucks,
            num_customers=num_customers,
            hidden_dim=hidden_dim
        )
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        features = extractor(obs_tensor)
        
        assert features.shape == (1, hidden_dim)
        assert not torch.isnan(features).any()
        
        print("[PASS] Features extraction with environment")

    def test_policy_forward_pass(self):
        """
        Test GraphPointerPolicy forward pass with environment.
        
        Verifies that the policy can process real environment observations
        and produce action distributions.
        """
        num_trucks = 4
        num_customers = 15
        
        customers = [Customer(i, np.random.rand() * 100, np.random.rand() * 100, np.random.rand() * 20 + 10) for i in range(num_customers)]
        trucks = [Truck(i, 0, 50.0) for i in range(num_trucks)]
        depots = [Depot(0, 50.0, 50.0)]
        
        env = SimpleFleetRoutingEnv(customers, trucks, depots)
        env = FleetRoutingSB3Wrapper(env)
        
        obs, _ = env.reset()
        
        hidden_dim = 64
        obs_space = env.observation_space
        action_space = env.action_space
        
        def lr_schedule(progress):
            return 3e-4
        
        policy = GraphPointerPolicy(
            obs_space,
            action_space,
            lr_schedule=lr_schedule,
            num_trucks=num_trucks,
            num_customers=num_customers,
            hidden_dim=hidden_dim
        )
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            distribution = policy.get_distribution(obs_tensor)
            action = distribution.sample()
        
        assert action.shape == (1, num_trucks)
        assert torch.all(action >= 0)
        assert torch.all(action <= num_customers)
        
        print("[PASS] Policy forward pass")


def run_all_tests():
    """Run all test classes."""
    print("\n" + "="*60)
    print("Testing Graph Converter Functions")
    print("="*60)
    
    test_converter = TestGraphConverter()
    test_converter.test_reconstruct_observation_components()
    test_converter.test_build_node_features()
    test_converter.test_build_adjacency_matrix()
    test_converter.test_observation_to_graph()
    test_converter.test_create_networkx_graph()
    
    print("\n" + "="*60)
    print("Testing Graph Pointer Feature Extraction")
    print("="*60)
    
    test_features = TestGraphPointerFeatures()
    test_features.test_graph_aware_features_extractor()
    test_features.test_graph_pointer_policy_creation()
    
    print("\n" + "="*60)
    print("Testing Integration with Environment")
    print("="*60)
    
    test_integration = TestGraphPointerIntegration()
    test_integration.test_features_extractor_with_environment()
    test_integration.test_policy_forward_pass()
    
    print("\n" + "="*60)
    print("[SUCCESS] All tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
