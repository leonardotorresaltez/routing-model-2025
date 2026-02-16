import matplotlib.pyplot as plt
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
    """
    Create a NetworkX directed graph representing the routing solution.
    Nodes are named using the original ID strings from the CSV
    (e.g. 'F1', 'C1', 'C2', ...).
    """

    G = nx.DiGraph()

    # Store lookup info inside the graph for later use
    G.graph["time_matrix"] = time_matrix
    G.graph["node_to_idx"] = node_to_idx
    G.graph["idx_to_node"] = idx_to_node
    G.graph["customers"] = customers

    # -----------------------------
    # Add depot nodes
    # -----------------------------
    for d in depots:
        node_id = d.id_str  # e.g. 'F1'
        G.add_node(
            node_id,
            pos=d.location(),
            type="depot",
            cluster=d.cluster_id,
        )

    # -----------------------------
    # Add customer nodes
    # -----------------------------
    for c in customers:
        node_id = c.id_str  # e.g. 'C1'
        G.add_node(
            node_id,
            pos=c.location(),
            type="customer",
            delivered=c.delivered,
            cluster=c.cluster_id,
        )

    # -----------------------------
    # Add edges based on routes
    # routes is a list of lists of node indices (idx)
    # truck_starts is a list of depot indices
    # -----------------------------
    for truck_id, route in enumerate(routes):
        if not route or len(route) <= 1:
            continue

        depot_idx = truck_starts[truck_id]
        depot_node = idx_to_node[depot_idx]  # e.g. 'F1'

        # Skip the first element if it's the depot index
        route_nodes = route[1:]

        # Depot -> first customer
        first_node = idx_to_node[route_nodes[0]]  # e.g. 'C1'
        G.add_edge(depot_node, first_node, truck=truck_id)

        # Customer -> customer
        for i in range(len(route_nodes) - 1):
            u = idx_to_node[route_nodes[i]]
            v = idx_to_node[route_nodes[i + 1]]
            G.add_edge(u, v, truck=truck_id)

        # Last customer -> depot
        last_node = idx_to_node[route_nodes[-1]]
        G.add_edge(last_node, depot_node, truck=truck_id)

    return G


def visualize_routing_solution(
    G: nx.DiGraph,
    step: int = 0,
    title_suffix: str = "",
    save_path: str = None,
):
    """
    Visualize the routing solution using the NetworkX graph and Matplotlib.
    Uses the real time_matrix to compute per-truck total time.
    """

    fig = plt.figure(figsize=(18, 10))
    ax_map = plt.subplot(121)
    ax_legend = plt.subplot(122)

    pos = nx.get_node_attributes(G, "pos")
    node_types = nx.get_node_attributes(G, "type")
    node_clusters = nx.get_node_attributes(G, "cluster")

    time_matrix = G.graph["time_matrix"]
    node_to_idx = G.graph["node_to_idx"]
    idx_to_node = G.graph["idx_to_node"]
    customers = G.graph["customers"]

    depot_nodes = [n for n, t in node_types.items() if t == "depot"]
    customer_nodes = [n for n, t in node_types.items() if t == "customer"]

    # -----------------------------
    # Draw depots
    # -----------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=depot_nodes,
        node_shape="*",
        node_size=500,
        node_color="gold",
        edgecolors="orange",
        linewidths=2,
        ax=ax_map,
        label="Depots",
    )

    # -----------------------------
    # Draw customers
    # -----------------------------
    delivered_nodes = [n for n in customer_nodes if G.nodes[n].get("delivered", False)]
    unvisited_nodes = [n for n in customer_nodes if not G.nodes[n].get("delivered", False)]

    if delivered_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=delivered_nodes,
            node_shape="o",
            node_size=150,
            node_color="#2ECC71",
            edgecolors="#27AE60",
            linewidths=1.5,
            ax=ax_map,
            label="Delivered",
        )

    if unvisited_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=unvisited_nodes,
            node_shape="o",
            node_size=150,
            node_color="#3498DB",
            edgecolors="#2C3E50",
            linewidths=1.5,
            ax=ax_map,
            label="Unvisited",
        )

    # -----------------------------
    # Draw edges
    # -----------------------------
    truck_colors = ["#E74C3C", "#9B59B6", "#F1C40F", "#1ABC9C", "#E67E22", "#34495E"]

    for u, v, data in G.edges(data=True):
        t_id = data["truck"]
        color = truck_colors[t_id % len(truck_colors)]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=2,
            alpha=0.6,
            edge_color=color,
            style="--",
            ax=ax_map,
            arrows=True,
            arrowsize=20,
        )

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax_map)

    ax_map.set_title(
        f"Step {step} - Map {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax_map.grid(True, alpha=0.3, linestyle="--")
    ax_map.legend(loc="upper right", fontsize=10)

    # ============================================================
    # LEGEND PANEL
    # ============================================================
    ax_legend.axis("off")
    legend_text = "TRUCK ASSIGNMENTS\n" + "=" * 30 + "\n\n"

    # -----------------------------
    # Reconstruct routes per truck
    # -----------------------------
    edge_trucks = nx.get_edge_attributes(G, "truck")
    truck_ids = sorted(set(edge_trucks.values()))
    truck_routes = {}

    for t_id in truck_ids:
        # Find starting depot for this truck
        start = next(
            (u for u, v, d in G.edges(data=True)
             if d["truck"] == t_id and node_types[u] == "depot"),
            None,
        )

        route_seq = []
        cur = start

        while True:
            nxt = next(
                (v for u, v, d in G.edges(data=True)
                 if u == cur and d["truck"] == t_id),
                None,
            )
            if nxt is None or node_types.get(nxt) == "depot":
                break
            route_seq.append(nxt)
            cur = nxt

        truck_routes[t_id] = route_seq

    # -----------------------------
    # Compute total time per truck (REAL TIME MATRIX)
    # -----------------------------
    truck_times = {}

    for t_id in truck_ids:
        total_time = 0.0
        for u, v, d in G.edges(data=True):
            if d["truck"] == t_id:
                i = node_to_idx[u]
                j = node_to_idx[v]
                total_time += float(time_matrix[i, j])
        truck_times[t_id] = total_time

    # -----------------------------
    # Add truck info to legend
    # -----------------------------
    for t_id in truck_ids:
        depot_label = next(
            (u for u, v, d in G.edges(data=True)
             if d["truck"] == t_id and node_types[u] == "depot"),
            "N/A",
        )

        legend_text += f"Truck {t_id}, Depot: {depot_label}\n"

        custs = truck_routes.get(t_id, [])
        if custs:
            full_route = [depot_label] + custs + [depot_label]
            legend_text += "   Route: " + " → ".join(full_route) + "\n"
        else:
            legend_text += "   Route: (empty)\n"

        legend_text += f"   Total Time: {truck_times[t_id]:.2f} h\n\n"

    # ============================================================
    # CLUSTER MEMBERSHIP LIST
    # ============================================================
    legend_text += "=" * 30 + "\nCLUSTERS\n"

    clusters = {}
    for c in customer_nodes:
        cid = node_clusters[c]
        clusters.setdefault(cid, []).append(c)

    for cid, nodes in clusters.items():
        legend_text += f"Cluster {cid}: " + ", ".join(nodes) + "\n"

    # ============================================================
    # DELIVERY SUMMARY
    # ============================================================
    delivered = [n for n in customer_nodes if G.nodes[n].get("delivered", False)]
    undelivered = [n for n in customer_nodes if not G.nodes[n].get("delivered", False)]

    legend_text += "\n" + "=" * 30 + "\n"
    legend_text += f"Delivered:   {len(delivered)}/{len(customer_nodes)}\n"
    legend_text += f"Undelivered: {len(undelivered)}/{len(customer_nodes)}\n"

    ax_legend.text(
        0.05,
        0.95,
        legend_text,
        transform=ax_legend.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round", facecolor="#ECF0F1", alpha=0.95),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()