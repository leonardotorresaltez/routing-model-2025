from copy import deepcopy


def evaluate_solution(tours,  data, truck_starts, cfg):
    # Check total time
    total_times = []
    full_tours = deepcopy(tours)
    # for i in range(len(full_tours)):
    #     full_tours[i].append(truck_starts[i])  # Ensure each tour ends at its depot
        # print(f"Tour {i} after appending depot: {full_tours[i]}")
    for truck_id, tour in enumerate(full_tours):
        
        total_time_truck = 0.0
        for i in range(len(tour)-1):
            from_node = tour[i]
            to_node = tour[i+1]
            travel_time = data["time_matrix"][from_node, to_node]
            total_time_truck += travel_time
        # total_time_truck += data["time_matrix"][truck_starts[truck_id], tour[0]]   # From depot to first
        # total_time_truck += data["time_matrix"][tour[-1], truck_starts[truck_id]]  # Return to depot
        
        if total_time_truck > cfg.max_daily_delivery_time_each_truck:
            raise ValueError(f"Truck {truck_id} exceeded max daily delivery time: {total_time_truck} > {cfg.max_daily_delivery_time_each_truck}")

        if tour[0] != truck_starts[truck_id]:
            raise ValueError(f"Truck {truck_id} did not start at its depot: started at {tour[0]}, should start at {truck_starts[truck_id]}")

        total_times.append(total_time_truck)

    # all_destinations_visited = [subtour for tour in full_tours for subtour in tour[1:]]
    all_destinations_visited = [subtour for tour in full_tours for subtour in tour[1:-1]]
    if len(set(all_destinations_visited)) != len(all_destinations_visited):
        raise ValueError("Some destinations were visited more than once!")
    
    total_destinations_visited = len(all_destinations_visited)
    total_time = sum(total_times).item()

    #print("full_tours:", full_tours)
    # print("Nodes: ", data["nodes"])
    full_tours_coords = [[(data["nodes"][node_idx].lat, data["nodes"][node_idx].lon) for node_idx in tour] for tour in full_tours]
    # print("full_tours coordinates:", full_tours_coords)

    def intersect(A, B, C, D):
        def ccw(P1, P2, P3):
            val = (P2[1] - P1[1]) * (P3[0] - P2[0]) - (P2[0] - P1[0]) * (P3[1] - P2[1])
            if val == 0: return 0
            return 1 if val > 0 else -1

        # General case
        return (ccw(A, B, C) != ccw(A, B, D) and 
                ccw(C, D, A) != ccw(C, D, B))

    total_checks = 0
    total_intersections = 0
    for truck_id, tour in enumerate(full_tours_coords):
        n_nodes = len(tour)
        for i in range(n_nodes-1):
            for j in range(i+2, n_nodes-1): # +1 to avoid himself, +1 more because it's impossible to intersect with adjacent edges
                if i == 0 and j == n_nodes-2:
                    continue  # Skip the case where we compare the first and last edge (both connected to depot)
                total_checks += 1
                # print(f"Checking edges ({i}, {i+1}) and ({j}, {j+1}) in truck {truck_id}")
                if intersect(tour[i], tour[i+1], tour[j], tour[j+1]):
                    # print(f"Intersection found between edges ({i}, {i+1}) and ({j}, {j+1}) in truck {truck_id}")
                    # print(f"Edge 1: {tour[i]} -> {tour[i+1]}, Edge 2: {tour[j]} -> {tour[j+1]}")
                    total_intersections += 1
    #print(f"Total edge pairs checked: {total_checks}, Total intersections found: {total_intersections}")
    # Avoid division by zero
    if total_checks == 0:
        pct_intersections = 0.0
    else:
        pct_intersections = total_intersections / total_checks
                    
    return total_destinations_visited, total_time, pct_intersections