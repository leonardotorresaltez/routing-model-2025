import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple
import os
import plotly.graph_objects as go

from core.utils.data_loader import Customer, Depot, Truck


def create_routing_graph(depots: List[Depot], customers: List[Customer], routes: dict, truck_starts: List[int]) -> nx.DiGraph:
    """Create a NetworkX directed graph representing the routing solution."""
    G = nx.DiGraph()
    
    # Add nodes and labels (types)
    for d in depots:
        G.add_node(f"D{d.idx}", pos=d.location(), type='depot')
    for c in customers:
        G.add_node(f"C{c.idx}", pos=c.location(), type='customer', delivered=c.delivered)
    
    # Add edges based on routes and annotate nodes with SEQUENCE
    for truck_id, route in enumerate(routes):
        if not route: continue
        
        # Determine home depot
        d_id = truck_starts[truck_id]
        
        # The route list usually includes the depot at index 0 or requires offset.
        # Based on your previous snippet: route = route[1:]
        # We assume the input 'route' is [Depot_ID, Cust_1, Cust_2, ...] or just [Cust_1, Cust_2...]
        # Adjusting logic to match your snippet:
        
        stops = route[1:] # Skip the first element (assumed to be depot or start marker)
        
        # 1. Depot -> First Customer
        if stops:
            G.add_edge(f"D{d_id}", f"C{stops[0]}", truck=truck_id)
        
        # 2. Customer -> Customer
        for i in range(len(stops)):
            current_node_id = f"C{stops[i]}"
            
            # --- NEW: Store Sequence Order on the Node ---
            # We use i+1 so the first customer is stop #1, not #0
            G.nodes[current_node_id]['seq'] = i + 1 
            G.nodes[current_node_id]['truck'] = truck_id
            
            # Add edge to next customer
            if i < len(stops) - 1:
                next_node_id = f"C{stops[i+1]}"
                G.add_edge(current_node_id, next_node_id, truck=truck_id)
            
        # 3. Last Customer -> Depot
        if stops:
            G.add_edge(f"C{stops[-1]}", f"D{d_id}", truck=truck_id)
        
    return G


def visualize_routing_solution(G: nx.DiGraph, step: int = 0, title_suffix: str = "", save_path: str = None):
    """
    Visualize the routing solution using Plotly with Sequence Numbers on Nodes.
    """
    pos = nx.get_node_attributes(G, 'pos')
    node_types = nx.get_node_attributes(G, 'type')
    
    # Initialize Figure
    fig = go.Figure()

    # --- 1. PLOT ROUTES (EDGES) ---
    truck_colors = ['#E74C3C', '#9B59B6', '#F1C40F', '#1ABC9C', '#E67E22', '#34495E', 
                    '#2ECC71', '#3498DB', '#95A5A6', '#D35400']
    
    edge_trucks = nx.get_edge_attributes(G, 'truck')
    unique_trucks = sorted(list(set(edge_trucks.values())))

    for t_id in unique_trucks:
        # Filter edges for this truck
        truck_edges = [(u, v) for (u, v), tid in edge_trucks.items() if tid == t_id]
        if not truck_edges: continue

        # Simple path reconstruction for plotting lines
        path_nodes = []
        starts = [u for u, v in truck_edges if u.startswith('D')]
        if starts:
            current = starts[0]
            path_nodes.append(current)
            edges_pool = list(truck_edges)
            while edges_pool:
                nxt_edge = next((e for e in edges_pool if e[0] == current), None)
                if nxt_edge:
                    current = nxt_edge[1]
                    path_nodes.append(current)
                    edges_pool.remove(nxt_edge)
                else:
                    break
        else:
            path_nodes = [n for edge in truck_edges for n in edge]

        # Extract coordinates
        edge_x = []
        edge_y = []
        for node in path_nodes:
            if node in pos:
                edge_x.append(pos[node][0])
                edge_y.append(pos[node][1])

        # Add Trace
        color = truck_colors[t_id % len(truck_colors)]
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
            name=f'Truck {t_id}',
            opacity=0.8,
            legendgroup=f'group_{t_id}'
        ))

    # --- 2. PLOT NODES WITH SEQUENCE NUMBERS ---
    
    def add_node_trace(node_list, color, symbol, size, label_prefix):
        if not node_list: return
        
        x_vals = [pos[n][0] for n in node_list]
        y_vals = [pos[n][1] for n in node_list]
        
        # --- MODIFIED TEXT GENERATION ---
        display_texts = []
        hover_texts = []
        
        for n in node_list:
            # Check if this node has a sequence number assigned
            seq = G.nodes[n].get('seq')
            truck = G.nodes[n].get('truck')
            
            if seq is not None:
                # If it has a sequence, show the number (e.g., "1", "2")
                # We also assume if it has a sequence, it belongs to a truck
                display_texts.append(str(seq))
                hover_texts.append(f"{label_prefix}: {n}<br>Truck: {truck}<br>Stop: #{seq}")
            else:
                # Fallback for unvisited or depots: show the Node ID
                display_texts.append(n)
                hover_texts.append(f"{label_prefix}: {n}")

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text', # 'text' enables the labels on top
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color='Black')),
            
            text=display_texts, # This is what appears ON the map
            textposition="top center", # Places text strictly above the marker
            textfont=dict(size=10, color="black", family="Arial Black"), # Bold font for readability
            
            hovertext=hover_texts, # This appears when you hover mouse
            hoverinfo="text",
            name=label_prefix
        ))

    # Identify node groups
    depots = [n for n, t in node_types.items() if t == 'depot']
    delivered = [n for n, t in node_types.items() if t == 'customer' and G.nodes[n].get('delivered', False)]
    unvisited = [n for n, t in node_types.items() if t == 'customer' and not G.nodes[n].get('delivered', False)]

    add_node_trace(depots, 'gold', 'star', 15, 'Depot')
    add_node_trace(delivered, '#2ECC71', 'circle', 12, 'Delivered')
    add_node_trace(unvisited, '#3498DB', 'circle', 10, 'Unvisited')

    # --- 3. DASHBOARD STATS ---
    total_cust = len(delivered) + len(unvisited)
    perc_deliv = (len(delivered) / total_cust * 100) if total_cust > 0 else 0
    
    stats_text = (
        f"<b>Step: {step}</b><br>"
        f"Delivered: {len(delivered)}/{total_cust} ({perc_deliv:.1f}%)<br>"
        f"Trucks Active: {len(unique_trucks)}"
    )

    fig.add_annotation(
        text=stats_text,
        align='left',
        showarrow=False,
        xref='paper', yref='paper',
        x=0.01, y=0.99,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        opacity=0.9
    )

    fig.update_layout(
        title=dict(text=f"Routing Solution {title_suffix}", x=0.5),
        showlegend=True,
        legend=dict(title="Legend", itemsizing='constant'),
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, title="X"),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, title="Y"),
        autosize=True, 
        plot_bgcolor='#F0F2F6',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    if save_path:
        if not save_path.endswith('.html'):
            save_path = save_path.rsplit('.', 1)[0] + '.html'
        fig.write_html(save_path, include_plotlyjs='cdn', full_html=True)
        print(f"Saved interactive visualization to {save_path}")
    else:
        fig.show(config={'responsive': True})