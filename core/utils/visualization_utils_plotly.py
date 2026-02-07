import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

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
        
        if route[0] == d_id:
            route = route[1:]
        if route and route[-1] == d_id:
            route = route[:-1]
        # # Depot -> First Customer
        # route = route[1:]  # Skip depot in route to add it in the next line with the correct prefix
        G.add_edge(f"D{d_id}", f"C{route[0]}", truck=truck_id)
        
        # Customer -> Customer
        for i in range(len(route) - 1):
            G.add_edge(f"C{route[i]}", f"C{route[i+1]}", truck=truck_id)
            
        # An last Customer -> Depot # FIXME
        G.add_edge(f"C{route[-1]}", f"D{d_id}", truck=truck_id)
    
        
    return G


def visualize_routing_solution(G: nx.DiGraph, step: int = 0, title_suffix: str = "", save_path: str = None):
    """
    Visualize the routing solution using Plotly (Interactive).
    """
    pos = nx.get_node_attributes(G, 'pos')
    node_types = nx.get_node_attributes(G, 'type')
    
    # Initialize Figure
    fig = go.Figure()

    # --- 1. PLOT ROUTES (EDGES) ---
    # We group edges by truck to create one continuous line trace per truck.
    # This allows toggling specific trucks in the legend.
    
    truck_colors = ['#E74C3C', '#9B59B6', '#F1C40F', '#1ABC9C', '#E67E22', '#34495E', 
                    '#2ECC71', '#3498DB', '#95A5A6', '#D35400']
    
    # Get all unique truck IDs present in the edges
    edge_trucks = nx.get_edge_attributes(G, 'truck')
    unique_trucks = sorted(list(set(edge_trucks.values())))

    for t_id in unique_trucks:
        # Filter edges for this truck
        truck_edges = [(u, v) for (u, v), tid in edge_trucks.items() if tid == t_id]
        
        if not truck_edges:
            continue

        # Sort edges to form a continuous path for plotting
        # Strategy: Find the start (Depot) and follow the chain
        # Note: This simple sorter assumes a single continuous path per truck
        path_nodes = []
        
        # Find the node that is a source but not a target within this truck's subgraph
        # Or simply start at the Depot
        starts = [u for u, v in truck_edges if u.startswith('D')]
        if starts:
            current = starts[0]
            path_nodes.append(current)
            
            # Simple greedy path reconstruction
            # (In complex graphs with cycles/branches, this needs a full traversal algorithm)
            edges_pool = list(truck_edges)
            while edges_pool:
                # Find edge starting from 'current'
                nxt_edge = next((e for e in edges_pool if e[0] == current), None)
                if nxt_edge:
                    current = nxt_edge[1]
                    path_nodes.append(current)
                    edges_pool.remove(nxt_edge)
                else:
                    break
        else:
            # Fallback if no depot start found (just draw segments)
            path_nodes = [n for edge in truck_edges for n in edge]

        # Extract coordinates
        edge_x = []
        edge_y = []
        for node in path_nodes:
            if node in pos:
                edge_x.append(pos[node][0])
                edge_y.append(pos[node][1])
            else:
                # Handle missing node key error gracefully
                print(f"Warning: Node {node} missing from positions.")

        # Add Trace
        color = truck_colors[t_id % len(truck_colors)]
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines+markers', # Markers help see direction implicitly
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
            name=f'Truck {t_id}',
            opacity=0.8,
            legendgroup=f'group_{t_id}'
        ))

    # --- 2. PLOT NODES ---
    
    # Helper to build node traces
    def add_node_trace(node_list, color, symbol, size, label_prefix):
        if not node_list: return
        
        x_vals = [pos[n][0] for n in node_list]
        y_vals = [pos[n][1] for n in node_list]
        hover_texts = [f"{label_prefix}: {n}" for n in node_list]
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color='Black')),
            text=node_list if len(node_list) < 50 else None, # Only show labels on map if few nodes
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_texts,
            hoverinfo="text",
            name=label_prefix
        ))

    # Identify node groups
    depots = [n for n, t in node_types.items() if t == 'depot']
    delivered = [n for n, t in node_types.items() if t == 'customer' and G.nodes[n].get('delivered', False)]
    unvisited = [n for n, t in node_types.items() if t == 'customer' and not G.nodes[n].get('delivered', False)]

    add_node_trace(depots, 'gold', 'star', 15, 'Depot')
    add_node_trace(delivered, '#2ECC71', 'circle', 10, 'Delivered')
    add_node_trace(unvisited, '#3498DB', 'circle', 10, 'Unvisited')

    # --- 3. DASHBOARD STATS (Annotation) ---
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
        
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, title="X Coordinate"),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, title="Y Coordinate"),
        
        autosize=True, 
        plot_bgcolor='#F0F2F6',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # OUTPUT
    if save_path:
        # Force the extension to .html to ensure interactivity and responsiveness
        if not save_path.endswith('.html'):
            save_path = save_path.rsplit('.', 1)[0] + '.html'
        
        # full_html=True makes it a standalone file you can open anywhere
        # include_plotlyjs='cdn' keeps the file size small
        fig.write_html(save_path, include_plotlyjs='cdn', full_html=True)
        print(f"Saved interactive visualization to {save_path}")
    else:
        # Shows in browser/notebook with responsive config
        fig.show(config={'responsive': True})