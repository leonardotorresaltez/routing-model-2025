import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from typing import List
from core.utils.data_loader import Customer, Depot

def create_routing_graph(
    depots: List[Depot],
    customers: List[Customer],
    routes: list,
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

    # Add depot nodes
    for d in depots:
        G.add_node(d.id_str, pos=d.location(), type="depot", cluster=d.cluster_id)

    # Add customer nodes
    for c in customers:
        G.add_node(c.id_str, pos=c.location(), type="customer", delivered=c.delivered, cluster=c.cluster_id)

    # Add edges based on routes
    for truck_id, route in enumerate(routes):
        if not route or len(route) <= 1:
            continue
        depot_node = idx_to_node[truck_starts[truck_id]]
        route_nodes = route[1:] # Skip first depot entry

        # Path sequence
        prev = depot_node
        for idx in route_nodes:
            curr = idx_to_node[idx]
            G.add_edge(prev, curr, truck=truck_id)
            prev = curr
        # Return to depot
        G.add_edge(prev, depot_node, truck=truck_id)

    return G

def visualize_routing_solution(G: nx.DiGraph, step: int = 0, title_suffix: str = "", save_path: str = None):
    # --- 1. SET UP THE DASHBOARD LAYOUT ---
    fig = plt.figure(figsize=(24, 12))
    # Map gets 3 parts of width, Legend gets 1 part
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    
    ax_map = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    
    pos = nx.get_node_attributes(G, "pos")
    node_types = nx.get_node_attributes(G, "type")
    time_matrix = G.graph["time_matrix"]
    node_to_idx = G.graph["node_to_idx"]

    # --- 2. DRAW MAP ELEMENTS ---
    depot_nodes = [n for n, t in node_types.items() if t == "depot"]
    delivered_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "customer" and G.nodes[n].get("delivered")]
    unvisited_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "customer" and not G.nodes[n].get("delivered")]

    # Depots (Gold Stars)
    nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, node_shape="*", node_size=600, 
                           node_color="gold", edgecolors="black", ax=ax_map, label="Depots")
    
    # Customers
    nx.draw_networkx_nodes(G, pos, nodelist=delivered_nodes, node_size=200, 
                           node_color="#2ECC71", edgecolors="#27AE60", ax=ax_map, label="Delivered")
    nx.draw_networkx_nodes(G, pos, nodelist=unvisited_nodes, node_size=200, 
                           node_color="#3498DB", edgecolors="#2C3E50", ax=ax_map, label="Unvisited")

    # Edges (Truck Routes)
    truck_colors = ["#E74C3C", "#9B59B6", "#F1C40F", "#1ABC9C", "#E67E22", "#34495E", "#8E44AD"]
    edge_trucks = nx.get_edge_attributes(G, "truck")
    
    for t_id in set(edge_trucks.values()):
        t_edges = [e for e, tid in edge_trucks.items() if tid == t_id]
        nx.draw_networkx_edges(G, pos, edgelist=t_edges, width=2.5, alpha=0.7, 
                               edge_color=truck_colors[t_id % len(truck_colors)], 
                               arrows=True, arrowsize=15, ax=ax_map)

    # Labels (Slightly smaller to avoid overlap)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax_map)

    ax_map.set_title(f"VRP Training Step {step} | {title_suffix}", fontsize=16, fontweight="bold")
    ax_map.set_facecolor("#F9F9F9")
    ax_map.grid(True, linestyle="--", alpha=0.5)
    ax_map.legend(loc="upper right")

    # --- 3. CONSTRUCT THE DATA LEGEND ---
    ax_legend.axis("off")
    summary_text = "TRUCK ASSIGNMENTS\n" + "="*25 + "\n"
    
    # Calculate route times
    for t_id in sorted(set(edge_trucks.values())):
        total_time = 0.0
        route_nodes = []
        # Find start depot
        curr = next(u for u, v, d in G.edges(data=True) if d["truck"] == t_id and node_types[u] == "depot")
        route_nodes.append(curr)
        
        # Follow the path
        for _ in range(len([e for e, tid in edge_trucks.items() if tid == t_id])):
            nxt = next((v for u, v, d in G.edges(data=True) if u == curr and d["truck"] == t_id), None)
            if nxt:
                total_time += float(time_matrix[node_to_idx[curr], node_to_idx[nxt]])
                route_nodes.append(nxt)
                curr = nxt
                if node_types[curr] == "depot": break

        summary_text += f"Truck {t_id}: {total_time:.2f}h\n"
        summary_text += f" -> {' -> '.join(route_nodes[:5])}..." if len(route_nodes) > 5 else f" -> {' -> '.join(route_nodes)}\n"
        summary_text += "\n"

    # Add Cluster Info
    summary_text += "="*25 + "\nCLUSTER STATUS\n"
    clusters = {}
    for n in G.nodes:
        if node_types[n] == "customer":
            cid = G.nodes[n].get("cluster", "N/A")
            clusters.setdefault(cid, []).append(n)
    
    for cid, nodes in sorted(clusters.items()):
        delivered_count = sum(1 for n in nodes if G.nodes[n].get("delivered"))
        summary_text += f"Cluster {cid}: {delivered_count}/{len(nodes)} Done\n"

    # Add terminal summary
    summary_text += "\n" + "="*25 + f"\nTOTAL DELIVERED: {len(delivered_nodes)}/{len(delivered_nodes)+len(unvisited_nodes)}"

    # Render text in a nice box
    ax_legend.text(0, 1, summary_text, transform=ax_legend.transAxes, fontsize=10,
                   verticalalignment="top", family="monospace", fontweight="bold",
                   bbox=dict(boxstyle="round", facecolor="#FDFEFE", edgecolor="#D5DBDB"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()