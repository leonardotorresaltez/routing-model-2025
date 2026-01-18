"""
Convert flat observations to graph representations for Graph Pointer Network.

This module extracts node features and adjacency matrices from flat observation
vectors, enabling the GraphPointerNetwork to process fleet routing problems as
explicit graph structures.
"""

import networkx as nx
import numpy as np
import torch


def reconstruct_observation_components(
    observation: np.ndarray, 
    num_trucks: int, 
    num_customers: int
) -> dict:
    """
    Reconstruct individual components from flat observation vector.
    
    Reverses the flattening process in sb3_wrapper.py to extract meaningful
    components from the concatenated observation vector.
    
    Args:
        observation: Flat 1D array of size 5*T + 4*N
        num_trucks: Number of trucks (T)
        num_customers: Number of customers (N)
    
    Returns:
        Dictionary with keys:
            - truck_positions: [T, 2] array of (x, y) coordinates
            - truck_loads: [T] array of current loads
            - truck_capacities: [T] array of max capacities
            - truck_utilization: [T] array of load/capacity ratios
            - customer_locations: [N, 2] array of (x, y) coordinates
            - customer_weights: [N] array of delivery demands
            - unvisited_mask: [N] array of visit status (1=unvisited, 0=visited)
    """
    T = num_trucks
    N = num_customers
    
    idx = 0
    
    truck_positions = observation[idx:idx+2*T].reshape(T, 2)
    idx += 2*T
    
    truck_loads = observation[idx:idx+T]
    idx += T
    
    truck_capacities = observation[idx:idx+T]
    idx += T
    
    truck_utilization = observation[idx:idx+T]
    idx += T
    
    customer_locations = observation[idx:idx+2*N].reshape(N, 2)
    idx += 2*N
    
    customer_weights = observation[idx:idx+N]
    idx += N
    
    unvisited_mask = observation[idx:idx+N]
    
    return {
        'truck_positions': truck_positions,
        'truck_loads': truck_loads,
        'truck_capacities': truck_capacities,
        'truck_utilization': truck_utilization,
        'customer_locations': customer_locations,
        'customer_weights': customer_weights,
        'unvisited_mask': unvisited_mask,
    }


def build_node_features(
    truck_positions: np.ndarray,
    customer_locations: np.ndarray
) -> torch.Tensor:
    """
    Build node feature matrix combining truck and customer positions.
    
    Concatenates truck and customer positions to create a unified node feature
    matrix where each row represents either a truck or customer node.
    
    Args:
        truck_positions: [T, 2] array of truck coordinates
        customer_locations: [N, 2] array of customer coordinates
    
    Returns:
        Tensor of shape [T+N, 2] with all node positions
    """
    all_positions = np.vstack([truck_positions, customer_locations])
    return torch.FloatTensor(all_positions)


def build_adjacency_matrix(
    node_features: torch.Tensor,
    k_neighbors: int = 5,
    threshold: float = None
) -> torch.Tensor:
    """
    Build adjacency matrix based on spatial proximity (k-nearest neighbors).
    
    Creates a sparse adjacency matrix where nodes are connected to their
    k nearest neighbors by Euclidean distance. This captures the spatial
    structure of the routing problem.
    
    Args:
        node_features: [N_nodes, 2] tensor of node positions
        k_neighbors: Number of nearest neighbors to connect (default: 5)
        threshold: Alternative to k_neighbors - connect if distance < threshold
                  If specified, k_neighbors is ignored
    
    Returns:
        Adjacency matrix of shape [N_nodes, N_nodes] with values {0, 1}
    """
    n_nodes = node_features.shape[0]
    
    distances = torch.cdist(node_features, node_features)
    adjacency = torch.zeros(n_nodes, n_nodes)
    
    if threshold is not None:
        adjacency = (distances < threshold).float()
    else:
        for i in range(n_nodes):
            sorted_distances, indices = torch.topk(
                distances[i], 
                k=min(k_neighbors + 1, n_nodes),
                largest=False
            )
            adjacency[i, indices] = 1.0
    
    symmetrize = (adjacency + adjacency.T) > 0
    adjacency = symmetrize.float()
    
    return adjacency # if specificied (it will zero the pairs of no interest (neighbours or out of threshold value))


def observation_to_graph(
    observation: np.ndarray,
    num_trucks: int,
    num_customers: int,
    k_neighbors: int = None
) -> tuple:
    """
    Convert flat observation to graph representation (node features, adjacency).
    
    This is the main conversion function that orchestrates the extraction and
    graph building process. It transforms the flat 80-dimensional observation
    into an explicit graph structure suitable for GraphPointerNetwork processing.
    
    Args:
        observation: Flat observation vector of size 5*T + 4*N
        num_trucks: Number of trucks
        num_customers: Number of customers
        k_neighbors: Number of nearest neighbors for adjacency (default: None for full connectivity)
    
    Returns:
        Tuple of (node_features, adjacency_matrix):
            - node_features: [T+N, 2] tensor of node positions
            - adjacency_matrix: [T+N, T+N] tensor representing graph structure
    """
    
    
    # Creating and using graphs correctly via:
    # observation_to_graph() (lines 140-181): Converts flat observations to graph tensors
    # build_node_features() (lines 76-94): Creates node feature matrix
    # build_adjacency_matrix() (lines 97-137): Creates connectivity pattern
    # Feed to GNN in SimpleGraphFeaturesExtractor.forward(): PyTorch processes them
    # Bottom Line
    # No changes needed. The graph exists as PyTorch tensors (most efficient for neural networks). create_networkx_graph() is optional utility for visualization—you don't need it for PPO training.

    components = reconstruct_observation_components(
        observation, num_trucks, num_customers
    )
    
    node_features = build_node_features(
        components['truck_positions'],
        components['customer_locations']
    )
    
    if k_neighbors is None:
        k_neighbors = num_trucks + num_customers
    
    adjacency_matrix = build_adjacency_matrix(
        node_features,
        k_neighbors=k_neighbors
    )
    
    return node_features, adjacency_matrix


def create_networkx_graph(
    node_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    node_attributes: dict = None
) -> nx.Graph:
    """
    Create NetworkX graph from node features and adjacency matrix.
    
    Useful for visualization and analysis of the routing graph structure.
    Can optionally attach node attributes like loads, capacities, demands.
    
    Args:
        node_features: [N_nodes, 2] tensor of node positions
        adjacency_matrix: [N_nodes, N_nodes] adjacency tensor
        node_attributes: Optional dict with node-level attributes
                        Keys should be attribute names, values should be lists
    
    Returns:
        NetworkX Graph object with nodes, edges, and optional attributes
    """
    n_nodes = node_features.shape[0]
    G = nx.Graph()
    
    for i in range(n_nodes):
        x, y = float(node_features[i, 0]), float(node_features[i, 1])
        G.add_node(i, pos=(x, y))
        
        if node_attributes:
            for attr_name, attr_values in node_attributes.items():
                if i < len(attr_values):
                    G.nodes[i][attr_name] = float(attr_values[i])
    
    adj_np = adjacency_matrix.cpu().numpy() if isinstance(adjacency_matrix, torch.Tensor) else adjacency_matrix
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_np[i, j] > 0:
                G.add_edge(i, j)
    
    return G


def visualize_routing_graph(
    node_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    num_trucks: int,
    num_customers: int,
    title: str = "Routing Graph",
    figsize: tuple = (10, 8),
    save_path: str = None
) -> None:
    """
    Visualize the routing graph with NetworkX and matplotlib.
    
    Visualizes trucks and customers as nodes with edges based on adjacency.
    Trucks are colored red, customers are colored blue. Node positions are
    based on actual (x, y) coordinates from node_features.
    
    Args:
        node_features: [T+N, 2] tensor of node positions
        adjacency_matrix: [T+N, T+N] tensor representing graph structure
        num_trucks: Number of trucks
        num_customers: Number of customers
        title: Title for the plot
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
    
    Example:
        >>> visualize_routing_graph(node_features, adjacency_matrix, 4, 15,
        ...                         title="Step 10", save_path="graphs/step_10.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install it with: pip install matplotlib")
        return
    
    G = create_networkx_graph(node_features, adjacency_matrix)
    
    pos = {}
    for node_id in G.nodes():
        x = float(node_features[node_id, 0])
        y = float(node_features[node_id, 1])
        pos[node_id] = (x, y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    truck_nodes = list(range(num_trucks))
    customer_nodes = list(range(num_trucks, num_trucks + num_customers))
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=truck_nodes,
        node_color='red',
        node_size=300,
        label='Trucks',
        ax=ax,
        alpha=0.9
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=customer_nodes,
        node_color='blue',
        node_size=200,
        label='Customers',
        ax=ax,
        alpha=0.7
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        width=0.5,
        alpha=0.3,
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_color='black',
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_routing_solution(
    observation: np.ndarray,
    num_trucks: int,
    num_customers: int,
    depots: list = None,
    truck_routes: dict = None,
    step: int = 0,
    title_suffix: str = "",
    save_path: str = None
) -> None:
    """
    Visualize routing solution with depots, trucks, and customer assignments.
    
    Clear visualization showing:
    - ⭐ Yellow Stars: Depots
    - 🔴 Red Squares: Trucks (labeled T0, T1, ...)
    - 🟢 Green Circles: Delivered customers (labeled C0, C1, ...)
    - 🔵 Blue Circles: Unvisited customers
    - Node size: Reflects truck load or customer weight
    - Legend: Shows each truck and its assigned customers
    
    Args:
        observation: Flat observation vector of size 5*T + 4*N
        num_trucks: Number of trucks
        num_customers: Number of customers
        depots: List of depot positions [(x, y), ...] or None
        truck_routes: Dict mapping truck_id -> [customer_ids] (optional)
        step: Step number (for naming)
        title_suffix: Additional text for the title
        save_path: Path to save the figure (if None, displays plot)
    
    Example:
        >>> depots = [(100, 100), (200, 200)]
        >>> truck_routes = {0: [1, 3, 5], 1: [2, 4], 2: [], 3: [6, 7, 8]}
        >>> visualize_routing_solution(obs, 4, 15, depots=depots,
        ...                            truck_routes=truck_routes, step=10,
        ...                            save_path="solution_10.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install it with: pip install matplotlib")
        return
    
    components = reconstruct_observation_components(
        observation, num_trucks, num_customers
    )
    
    truck_positions = components['truck_positions']
    customer_locations = components['customer_locations']
    truck_loads = components['truck_loads']
    truck_capacities = components['truck_capacities']
    customer_weights = components['customer_weights']
    unvisited_mask = components['unvisited_mask']
    
    fig = plt.figure(figsize=(18, 10))
    ax_map = plt.subplot(121)
    ax_legend = plt.subplot(122)
    
    truck_node_sizes = 300 + 400 * (truck_loads / (truck_capacities + 1e-8))
    customer_weights_normalized = customer_weights / (customer_weights.max() + 1e-8) * 250 + 100
    
    delivered_mask = unvisited_mask == 0
    unvisited_indices = np.where(~delivered_mask)[0]
    delivered_indices = np.where(delivered_mask)[0]
    
    truck_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    truck_positions_scattered = truck_positions.copy()
    unique_positions = {}
    for i, pos in enumerate(truck_positions):
        pos_key = tuple(np.round(pos, 2))
        if pos_key not in unique_positions:
            unique_positions[pos_key] = []
        unique_positions[pos_key].append(i)
    
    for pos_key, truck_ids in unique_positions.items():
        if len(truck_ids) > 1:
            angle_step = 360 / len(truck_ids)
            radius = 0.5
            for idx, truck_id in enumerate(truck_ids):
                angle = np.radians(idx * angle_step)
                offset_x = radius * np.cos(angle)
                offset_y = radius * np.sin(angle)
                truck_positions_scattered[truck_id] = truck_positions[truck_id] + np.array([offset_x, offset_y])
    
    ax_map.scatter(
        truck_positions_scattered[:, 0],
        truck_positions_scattered[:, 1],
        s=truck_node_sizes,
        c='red',
        marker='s',
        alpha=0.85,
        edgecolors='darkred',
        linewidth=2.5,
        zorder=3,
        label='Trucks'
    )
    
    if depots is not None:
        depots = np.array(depots)
        ax_map.scatter(
            depots[:, 0],
            depots[:, 1],
            s=400,
            c='gold',
            marker='*',
            alpha=0.9,
            edgecolors='orange',
            linewidth=2,
            zorder=4,
            label='Depots'
        )
        for i, (x, y) in enumerate(depots):
            ax_map.annotate(f'D{i}', (x, y), xytext=(8, 8), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='darkorange')
    
    if len(unvisited_indices) > 0:
        ax_map.scatter(
            customer_locations[unvisited_indices, 0],
            customer_locations[unvisited_indices, 1],
            s=customer_weights_normalized[unvisited_indices],
            c='#3498DB',
            marker='o',
            alpha=0.7,
            edgecolors='#2C3E50',
            linewidth=1.5,
            zorder=2,
            label='Unvisited Customers'
        )
    
    if len(delivered_indices) > 0:
        ax_map.scatter(
            customer_locations[delivered_indices, 0],
            customer_locations[delivered_indices, 1],
            s=customer_weights_normalized[delivered_indices],
            c='#2ECC71',
            marker='o',
            alpha=0.75,
            edgecolors='#27AE60',
            linewidth=1.5,
            zorder=2,
            label='Delivered Customers'
        )
    
    for i, (x, y) in enumerate(truck_positions_scattered):
        ax_map.annotate(f'T{i}', (x, y), xytext=(0, 0), textcoords='offset points',
                       fontsize=11, fontweight='bold', color='white', ha='center', va='center')
    
    for i, (x, y) in enumerate(customer_locations):
        status = '✓' if delivered_mask[i] else '○'
        ax_map.annotate(f'C{i}{status}', (x, y), xytext=(6, 6), textcoords='offset points',
                       fontsize=7, color='black', fontweight='bold')
    
    ax_map.set_title(f"Step {step} - Fleet Routing Map", fontsize=14, fontweight='bold')
    ax_map.set_xlabel('X Coordinate', fontsize=11)
    ax_map.set_ylabel('Y Coordinate', fontsize=11)
    ax_map.grid(True, alpha=0.3, linestyle='--')
    ax_map.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    ax_legend.axis('off')
    
    legend_y = 0.98
    legend_text = "TRUCK ASSIGNMENTS\n" + "="*50 + "\n\n"
    
    for truck_id in range(num_trucks):
        load_percent = (truck_loads[truck_id] / truck_capacities[truck_id]) * 100
        legend_text += f"🚚 Truck T{truck_id}:\n"
        legend_text += f"   Load: {truck_loads[truck_id]:.0f}/{truck_capacities[truck_id]:.0f} kg ({load_percent:.1f}%)\n"
        
        if truck_routes and truck_id in truck_routes:
            customers = truck_routes[truck_id]
            if customers:
                legend_text += f"   Customers: {', '.join([f'C{c}' for c in customers])}\n"
            else:
                legend_text += f"   Customers: (empty)\n"
        else:
            legend_text += f"   Customers: (tracking not available)\n"
        
        legend_text += "\n"
    
    legend_text += "="*50 + "\n\n"
    legend_text += "DELIVERY STATUS\n"
    legend_text += f"✓ Delivered: {len(delivered_indices)}/{num_customers}\n"
    legend_text += f"○ Unvisited: {len(unvisited_indices)}/{num_customers}\n"
    
    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                  fontsize=11, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.95, pad=1.5, linewidth=2),
                  fontweight='bold', color='#2C3E50')
    
    fig.suptitle(f"Routing Solution - {title_suffix}" if title_suffix else "Routing Solution",
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Solution visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_routing_step(
    observation: np.ndarray,
    num_trucks: int,
    num_customers: int,
    step: int = 0,
    title_suffix: str = "",
    save_dir: str = None
) -> None:
    """
    Convert observation to graph and visualize it.
    
    Convenience function that combines observation_to_graph and visualization
    for easier step-by-step debugging during training.
    
    Args:
        observation: Flat observation vector of size 5*T + 4*N
        num_trucks: Number of trucks
        num_customers: Number of customers
        step: Step number (for naming)
        title_suffix: Additional text for the title
        save_dir: Directory to save visualizations (if None, displays plot)
    
    Example:
        >>> for step in range(100):
        ...     obs, info = env.step(action)
        ...     if step % 10 == 0:
        ...         visualize_routing_step(obs, 4, 15, step, save_dir="debug_graphs/")
    """
    node_features, adjacency_matrix = observation_to_graph(
        observation, num_trucks, num_customers
    )
    
    title = f"Step {step}"
    if title_suffix:
        title += f" - {title_suffix}"
    
    save_path = None
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:04d}.png")
    
    visualize_routing_graph(
        node_features, adjacency_matrix,
        num_trucks, num_customers,
        title=title,
        save_path=save_path
    )
