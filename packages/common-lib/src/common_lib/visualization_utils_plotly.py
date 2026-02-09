import networkx as nx
import plotly.graph_objects as go
from typing import List

# Assuming these are your custom classes
from loader_lib.data_loader import Customer, Depot, Truck

def create_routing_graph(depots: List[Depot], customers: List[Customer], routes: dict, truck_starts: List[int]) -> nx.DiGraph:
    """Create a NetworkX directed graph with sequence attributes for animation."""
    G = nx.DiGraph()
    
    # 1. Add Nodes
    for d in depots:
        G.add_node(f"D{d.idx}", pos=d.location(), type='depot', seq=0)
    for c in customers:
        G.add_node(f"C{c.idx}", pos=c.location(), type='customer', delivered=c.delivered)
    
    # 2. Add Edges & Annotate Sequence
    for truck_id, route in enumerate(routes):
        if not route: continue
        d_id = truck_starts[truck_id]
        stops = route[1:] 
        
        # Depot -> First Customer
        if stops:
            G.add_edge(f"D{d_id}", f"C{stops[0]}", truck=truck_id, edge_seq=1)
        
        # Customer -> Customer
        for i in range(len(stops)):
            current_node = f"C{stops[i]}"
            seq_num = i + 1
            G.nodes[current_node]['seq'] = seq_num
            G.nodes[current_node]['truck'] = truck_id
            
            if i < len(stops) - 1:
                next_node = f"C{stops[i+1]}"
                G.add_edge(current_node, next_node, truck=truck_id, edge_seq=seq_num + 1)
            
        # Last Customer -> Depot
        if stops:
            last_seq = len(stops)
            G.add_edge(f"C{stops[-1]}", f"D{d_id}", truck=truck_id, edge_seq=last_seq + 1)
        
    return G


def visualize_routing_solution(G: nx.DiGraph, step, title_suffix: str = "", save_path: str = None):
    """
    Visualize with a Manual Slider only (No Play button, No Axis Labels).
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    # --- 1. CALCULATE FIXED MAP BOUNDS ---
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_padding = (max(all_x) - min(all_x)) * 0.05
    y_padding = (max(all_y) - min(all_y)) * 0.05
    x_range = [min(all_x) - x_padding, max(all_x) + x_padding]
    y_range = [min(all_y) - y_padding, max(all_y) + y_padding]

    # --- 2. PRE-PROCESS PATHS ---
    truck_colors = ['#E74C3C', '#9B59B6', '#F1C40F', '#1ABC9C', '#E67E22', '#34495E', 
                    '#2ECC71', '#3498DB', '#95A5A6', '#D35400']
    
    edge_trucks = nx.get_edge_attributes(G, 'truck')
    unique_trucks = sorted(list(set(edge_trucks.values())))
    
    truck_paths = {}
    max_seq_found = 0

    for t_id in unique_trucks:
        t_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data.get('truck') == t_id]
        if not t_edges: continue

        t_edges.sort(key=lambda x: x[2].get('edge_seq', 0))
        
        start_node = t_edges[0][0]
        x_vals = [pos[start_node][0]]
        y_vals = [pos[start_node][1]]
        seqs = [0]
        
        current_seq = 0
        for u, v, data in t_edges:
            x_vals.append(pos[v][0])
            y_vals.append(pos[v][1])
            s = data.get('edge_seq', current_seq + 1)
            seqs.append(s)
            current_seq = s
            if s > max_seq_found: max_seq_found = s
            
        truck_paths[t_id] = {'x': x_vals, 'y': y_vals, 'seqs': seqs, 'color': truck_colors[t_id % len(truck_colors)]}

    # --- 3. PREPARE CUSTOMER LIST (STATIC ORDER) ---
    customer_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'customer']
    customer_nodes.sort() 
    
    cust_x = [pos[n][0] for n in customer_nodes]
    cust_y = [pos[n][1] for n in customer_nodes]
    cust_seqs = [G.nodes[n].get('seq', 999999) for n in customer_nodes]

    # --- 4. INITIALIZE FIGURE ---
    fig = go.Figure()

    # A. Trucks (Placeholders)
    for t_id in unique_trucks:
        t_data = truck_paths[t_id]
        fig.add_trace(go.Scatter(
            x=[t_data['x'][0]], y=[t_data['y'][0]], 
            mode='lines+markers',
            line=dict(width=3, color=t_data['color']),
            marker=dict(size=6, color=t_data['color']),
            name=f'Truck {t_id}'
        ))

    # B. Depots (Static Trace)
    depots = [n for n, d in G.nodes(data=True) if d['type'] == 'depot']
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in depots], y=[pos[n][1] for n in depots],
        mode='markers+text',
        marker=dict(symbol='star', size=15, color='gold', line=dict(width=1, color='black')),
        text=[n for n in depots], textposition="top center",
        name='Depot'
    ))

    # C. ALL Customers (Single Dynamic Trace)
    fig.add_trace(go.Scatter(
        x=cust_x, y=cust_y,
        mode='markers+text',
        marker=dict(
            size=12, 
            line=dict(width=1, color='black'),
            color=['#3498DB'] * len(customer_nodes)
        ),
        text=[''] * len(customer_nodes),
        textfont=dict(color='black', size=10, family='Arial Black'),
        name='Customers'
    ))

    # --- 5. CREATE FRAMES ---
    frames = []
    COLOR_DELIVERED = '#2ECC71' # Green
    COLOR_UNVISITED = '#3498DB' # Blue
    
    for step in range(1, max_seq_found + 1):
        frame_data = []
        
        # 1. Update Trucks
        for t_id in unique_trucks:
            path = truck_paths[t_id]
            filtered = [(x, y) for x, y, s in zip(path['x'], path['y'], path['seqs']) if s <= step]
            if filtered:
                fx, fy = zip(*filtered)
                frame_data.append(go.Scatter(x=fx, y=fy))
            else:
                frame_data.append(go.Scatter(x=[], y=[]))
        
        # 2. Depot (Unchanged)
        frame_data.append(go.Scatter()) 
        
        # 3. Update Customers
        current_colors = []
        current_texts = []
        
        for seq in cust_seqs:
            if seq <= step:
                current_colors.append(COLOR_DELIVERED)
                current_texts.append(str(seq))
            else:
                current_colors.append(COLOR_UNVISITED)
                current_texts.append("")

        frame_data.append(go.Scatter(marker=dict(color=current_colors), text=current_texts))
        
        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

    # --- 6. LAYOUT (CLEANED) ---
    slider_steps = []
    for step in range(1, max_seq_found + 1):
        slider_steps.append(dict(
            method="animate",
            args=[[str(step)], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=0))],
            label=str(step)
        ))

    fig.update_layout(
        title=f"Routing Progression {title_suffix}",
        
        # REMOVED TITLES HERE
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, range=x_range),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, range=y_range),
        
        plot_bgcolor='#F0F2F6',
        
        # REMOVED UPDATEMENUS (Play Button)
        
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Step: "},
            pad={"t": 50},
            steps=slider_steps
        )]
    )

    if save_path:
        if not save_path.endswith('.html'): save_path += '.html'
        fig.write_html(save_path, include_plotlyjs='cdn', full_html=True)
        print(f"Saved interactive animation to {save_path}")
    else:
        fig.show()