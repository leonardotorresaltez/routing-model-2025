import math

def get_curriculum_iterator(start_nodes, n_total_nodes, ratio_trucks_nodes):
    """
    Yields (n_nodes, n_trucks) for curriculum learning.
    """
    n_nodes = start_nodes
    
    while True:
        # Calculate trucks using ceiling
        n_trucks = math.ceil(n_nodes * ratio_trucks_nodes)
        
        # Yield the current state of the curriculum
        yield n_nodes, n_trucks
        
        # Increment nodes for the *next* time next() is called, 
        # capped at n_total_nodes
        if n_nodes < n_total_nodes:
            n_nodes += 1