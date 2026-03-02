import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from typing import List
from core.utils.data_loader import Customer, Depot

def create_routing_graph(
    depots: List[Depot],
    customers: List[Customer],
    tours: list,
    truck_starts: List[int],
    node_to_idx: dict,
    idx_to_node: dict,
    time_matrix
) -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph["time_matrix"] = time_matrix
    G.graph["node_to_idx"] = node_to_idx
    G.graph["idx_to_node"] = idx_to_node
    G.graph["customers"] = customers

    # Add depot nodes (Removed cluster)
    for d in depots:
        G.add_node(d.id_str, pos=d.location(), type="depot")
    visited_customer_indices = set()
    depot_indices_set = {node_to_idx[d.id_str] for d in depots}
    for tour in tours:
        for node_idx in tour:
            if node_idx not in depot_indices_set:
                visited_customer_indices.add(node_idx)
    # Add customer nodes (Removed cluster)
    for c in customers:
        G.add_node(c.id_str, pos=c.location(), type="customer", delivered=c.delivered)

    # Add edges based on routes
    for truck_id, route in enumerate(tours):
        if not route or len(route) <= 1:
            continue
            
        depot_node = idx_to_node[truck_starts[truck_id]]
        # The route includes the starting depot at index 0
        route_nodes = [idx_to_node[idx] for idx in route]

        # Path sequence: Depot -> Customer 1 -> ... -> Depot
        for i in range(len(route_nodes) - 1):
            G.add_edge(route_nodes[i], route_nodes[i+1], truck=truck_id)
            
        # Ensure it returns to the specific truck's start depot if not already there
        if route_nodes[-1] != depot_node:
            G.add_edge(route_nodes[-1], depot_node, truck=truck_id)

    return G

def visualize_routing_solution(G: nx.DiGraph, step: int = 0, title_suffix: str = "", save_path: str = None):
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    
    ax_map = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    
    pos = nx.get_node_attributes(G, "pos")
    node_types = nx.get_node_attributes(G, "type")
    time_matrix = G.graph["time_matrix"]
    node_to_idx = G.graph["node_to_idx"]

    # --- DRAW MAP ELEMENTS ---
    depot_nodes = [n for n, t in node_types.items() if t == "depot"]
    delivered_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "customer" and G.nodes[n].get("delivered")]
    unvisited_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "customer" and not G.nodes[n].get("delivered")]

    # Depots (Gold Stars)
    nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, node_shape="*", node_size=600, 
                           node_color="gold", edgecolors="black", ax=ax_map, label="Depots")
    
    # Customers (Delivered vs Unvisited)
    nx.draw_networkx_nodes(G, pos, nodelist=delivered_nodes, node_size=200, 
                           node_color="#2ECC71", edgecolors="#27AE60", ax=ax_map, label="Delivered")
    nx.draw_networkx_nodes(G, pos, nodelist=unvisited_nodes, node_size=200, 
                           node_color="#3498DB", edgecolors="#2C3E50", ax=ax_map, label="Unvisited")

    # Edges (Truck Routes)
    truck_colors = ["#E74C3C", "#9B59B6", "#F1C40F", "#1ABC9C", "#E67E22", "#34495E", "#8E44AD"]
    edge_trucks = nx.get_edge_attributes(G, "truck")
    
    for t_id in sorted(set(edge_trucks.values())):
        t_edges = [e for e, tid in edge_trucks.items() if tid == t_id]
        nx.draw_networkx_edges(G, pos, edgelist=t_edges, width=2.5, alpha=0.7, 
                               edge_color=truck_colors[t_id % len(truck_colors)], 
                               arrows=True, arrowsize=15, ax=ax_map)

    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax_map)

    ax_map.set_title(f"VRP Training Step {step} | {title_suffix}", fontsize=16, fontweight="bold")
    ax_map.set_facecolor("#F9F9F9")
    ax_map.grid(True, linestyle="--", alpha=0.5)
    ax_map.legend(loc="upper right")

    # --- CONSTRUCT THE DATA LEGEND (Cluster-free) ---
    ax_legend.axis("off")
    summary_text = "TRUCK ASSIGNMENTS\n" + "="*25 + "\n"
    
    for t_id in sorted(set(edge_trucks.values())):
        total_time = 0.0
        route_seq = []
        
        # Simple extraction of edges for this truck
        t_edges = [(u, v) for (u, v), tid in edge_trucks.items() if tid == t_id]
        
        # Calculate time (assuming route is stored in edge data)
        for u, v in t_edges:
            total_time += float(time_matrix[node_to_idx[u], node_to_idx[v]])
            if u not in route_seq: route_seq.append(u)
            if v not in route_seq: route_seq.append(v)

        summary_text += f"Truck {t_id}: {total_time:.2f}h\n"
        summary_text += f" -> {' -> '.join(route_seq[:4])}...\n\n"

    # Final Progress Summary
    total_cust = len(delivered_nodes) + len(unvisited_nodes)
    summary_text += "="*25 + f"\nTOTAL PROGRESS\n"
    summary_text += f"Delivered: {len(delivered_nodes)} / {total_cust}\n"
    summary_text += f"Completion: {(len(delivered_nodes)/total_cust)*100:.1f}%"

    ax_legend.text(0, 1, summary_text, transform=ax_legend.transAxes, fontsize=10,
                   verticalalignment="top", family="monospace", fontweight="bold",
                   bbox=dict(boxstyle="round", facecolor="#FDFEFE", edgecolor="#D5DBDB"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()