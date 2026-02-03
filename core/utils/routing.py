import torch


def apply_2opt(route, depot_idx, time_matrix):
    """
    Applies 2-opt local search to a single route.
    Goal: Iteratively swaps edges to remove self-intersections (crossings).
    
    Args:
        route: List of customer indices.
        depot_idx: The starting/ending depot for this truck.
        time_matrix: Tensor of travel times/distances between all nodes.
    """
    if len(route) < 2:
        return route
    
    # We add the depot at the start and end to optimize the full journey
    # [Depot, C1, C2, C3, Depot]
    best_route = [depot_idx] + list(route) + [depot_idx]
    improved = True
    
    while improved:
        improved = False
        # We don't swap the depot at index 0 or the last index
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                
                # Current edges: (i-1 -> i) and (j -> j+1)
                # Potential edges: (i-1 -> j) and (i -> j+1)
                
                old_dist = time_matrix[best_route[i-1], best_route[i]].item() + \
                           time_matrix[best_route[j], best_route[j+1]].item()
                
                new_dist = time_matrix[best_route[i-1], best_route[j]].item() + \
                           time_matrix[best_route[i], best_route[j+1]].item()
                
                if new_dist < old_dist:
                    # Reverse the segment between i and j
                    best_route[i:j+1] = list(reversed(best_route[i:j+1]))
                    improved = True
        
    # Return only the customers (remove the added depots)
    return best_route[1:-1]



def calculate_route_time(route, depot_idx, time_matrix):
    """Calculates total travel time for a sequence of nodes starting and ending at depot."""
    if not route:
        return 0.0
    
    total_time = 0.0
    prev = depot_idx
    
    for node in route:
        total_time += time_matrix[prev, node].item()
        prev = node
        
    # Add return to depot
    total_time += time_matrix[prev, depot_idx].item()
    return total_time
