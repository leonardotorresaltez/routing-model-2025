import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple

from core.utils.data_loader import Customer, Depot, Truck

def create_routing_graph(depots: List[Depot], customers: List[Customer], routes: dict, truck_starts: List[int]) -> nx.DiGraph:
    """Create a NetworkX directed graph representing the routing solution as suggested by Jorge"""
    G = nx.DiGraph()
    
    # Add nodes and labels (types)
    for d in depots:
        G.add_node(f"D{d.idx}", pos=d.location(), type='depot')
    for c in customers:
        G.add_node(f"C{c.idx}", pos=c.location(), type='customer', delivered=c.delivered)
    
    # print("G nodes:", G.nodes())

    # Add edges based on routes
    for truck_id, route in enumerate(routes):
        if not route: continue
        
        # Determine home depot
        # Dummy data, we assume truck i starts at depot i % num_depots
        d_id = truck_starts[truck_id]
        
        # # Depot -> First Customer
        route = route[1:]  # Skip depot in route to add it in the next line with the correct prefix
        G.add_edge(f"D{d_id}", f"C{route[0]}", truck=truck_id)
        
        # Customer -> Customer
        for i in range(len(route) - 1):
            G.add_edge(f"C{route[i]}", f"C{route[i+1]}", truck=truck_id)
            
        # An last Customer -> Depot # FIXME
        G.add_edge(f"C{route[-1]}", f"D{d_id}", truck=truck_id)
    
        
    return G


def visualize_routing_solution(G: nx.DiGraph, step: int = 0, title_suffix: str = "", save_path: str = None):
    """Visualize the routing solution using the NetworkX graph and Matplotlib."""
    fig = plt.figure(figsize=(18, 10))
    ax_map = plt.subplot(121)
    ax_legend = plt.subplot(122)
    
    pos = nx.get_node_attributes(G, 'pos')
    node_types = nx.get_node_attributes(G, 'type')
    
    # Extract node lists
    depot_nodes = [n for n, t in node_types.items() if t == 'depot']
    customer_nodes = [n for n, t in node_types.items() if t == 'customer']
    
    # Plot Depots
    nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, node_shape='*', node_size=500, 
                           node_color='gold', edgecolors='orange', linewidths=2, ax=ax_map, label='Depots')
    
    # Plot Customers (Delivered vs Unvisited)
    delivered_nodes = [n for n in customer_nodes if G.nodes[n].get('delivered', False)]
    unvisited_nodes = [n for n in customer_nodes if not G.nodes[n].get('delivered', False)]
    
    if delivered_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=delivered_nodes, node_shape='o', node_size=150, 
                               node_color='#2ECC71', edgecolors='#27AE60', linewidths=1.5, ax=ax_map, label='Delivered')
    
    if unvisited_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=unvisited_nodes, node_shape='o', node_size=150, 
                               node_color='#3498DB', edgecolors='#2C3E50', linewidths=1.5, ax=ax_map, label='Unvisited')
    
    # Plot Edges (Routes)
    truck_colors = ['#E74C3C', '#9B59B6', '#F1C40F', '#1ABC9C', '#E67E22', '#34495E']
    for u, v, data in G.edges(data=True):
        t_id = data['truck']
        color = truck_colors[t_id % len(truck_colors)]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, alpha=0.6, 
                               edge_color=color, style='--', ax=ax_map, arrows=True, arrowsize=20)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax_map)
    
    ax_map.set_title(f"Step {step} - Map (NetworkX Graph Representation)", fontsize=14, fontweight='bold')
    ax_map.grid(True, alpha=0.3, linestyle='--')
    ax_map.legend(loc='upper right', fontsize=10)
    
    # Legend text
    ax_legend.axis('off')
    l_text = "TRUCK ASSIGNMENTS\n" + "="*30 + "\n\n"
    
    # Group edges by truck for the legend and respect delivery order
    truck_routes = {}
    truck_ids = set(nx.get_edge_attributes(G, 'truck').values())
    
    for t_id in truck_ids:
        # Find the starting depot for this truck
        current_node = next((u for u, v, d in G.edges(data=True) if d['truck'] == t_id and u.startswith('D')), None)
        
        route_seq = []
        visited = set()
        while current_node and current_node not in visited:
            visited.add(current_node)
            # Find the next edge for this specific truck
            edge = next(((u, v) for u, v, d in G.edges(data=True) if u == current_node and d['truck'] == t_id), None)
            if not edge: break
            
            next_node = edge[1]
            if next_node.startswith('D'): break # Returned to depot
            
            route_seq.append(next_node)
            current_node = next_node
            
        truck_routes[t_id] = route_seq

    # Find total trucks from edges or indices
    max_truck = max([data['truck'] for u, v, data in G.edges(data=True)]) if G.edges else -1
    for t_id in range(max_truck + 1):
        # Infer depot from the first edge starting with D
        d_id = "N/A"
        for u, v, data in G.edges(data=True):
            if data['truck'] == t_id and u.startswith('D'):
                d_id = u
                break
        
        l_text += f"Truck: {t_id}, Depot: {d_id}:\n"
        custs = truck_routes.get(t_id, [])
        if custs:
            l_text += f"   Cust: {', '.join(custs)}\n"
        else:
            l_text += "   Cust: (empty)\n"
        l_text += "\n"
    
    # Add Delivery Status Summary
    l_text += "="*30 + "\n"
    total_cust = len(customer_nodes)
    num_deliv = len(delivered_nodes)
    num_undeliv = len(unvisited_nodes)
    perc_deliv = (num_deliv / total_cust) * 100 if total_cust > 0 else 0
    
    l_text += "DELIVERY STATUS\n"
    l_text += f"Delivered:   {num_deliv}/{total_cust} ({perc_deliv:.1f}%)\n"
    l_text += f"Undelivered: {num_undeliv}/{total_cust} ({100-perc_deliv:.1f}%)\n"
    
    ax_legend.text(0.05, 0.95, l_text, transform=ax_legend.transAxes, fontsize=11, 
                  verticalalignment='top', family='monospace', fontweight='bold', 
                  color='#2C3E50', bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.95))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()