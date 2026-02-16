from copy import deepcopy
import torch
def evaluate_solution(env, data, truck_starts, cfg):
   

    time_matrix = data["time_matrix"]  
    num_trucks = len(env.tours)

    #check for end at depots
    full_tours = deepcopy(env.tours)
    for i in range(num_trucks):
        if full_tours[i][-1] != truck_starts[i]:
            full_tours[i].append(truck_starts[i])

    total_times = []
    per_truck_ok = []

    #Per truck time
    for truck_id, tour in enumerate(full_tours):
        # Must start at its depot
        if tour[0] != truck_starts[truck_id]:
            raise ValueError(
                f"Truck {truck_id} did not start at its depot: "
                f"started at {tour[0]}, should start at {truck_starts[truck_id]}"
            )

        total_time_truck = 0.0
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            travel_time = float(time_matrix[from_node, to_node].item())
            total_time_truck += travel_time

        total_times.append(total_time_truck)
        per_truck_ok.append(total_time_truck <= cfg.max_daily_delivery_time_each_truck)


    all_destinations_visited = [
        node for tour in full_tours for node in tour[1:-1]
    ]

    if len(set(all_destinations_visited)) != len(all_destinations_visited):
        raise ValueError("Some destinations were visited more than once!")

    total_destinations_visited = len(all_destinations_visited)
    total_time = sum(total_times)

    nodes = data["nodes"]
    full_tours_coords = [
        [(nodes[node_idx].lat, nodes[node_idx].lon) for node_idx in tour]
        for tour in full_tours
    ]

    def intersect(A, B, C, D):
        def ccw(P1, P2, P3):
            val = (P2[1] - P1[1]) * (P3[0] - P2[0]) - (P2[0] - P1[0]) * (P3[1] - P2[1])
            if val == 0:
                return 0
            return 1 if val > 0 else -1

        return (ccw(A, B, C) != ccw(A, B, D) and
                ccw(C, D, A) != ccw(C, D, B))

    total_checks = 0
    total_intersections = 0
    for truck_id, tour in enumerate(full_tours_coords):
        n_nodes = len(tour)
        for i in range(n_nodes - 1):
            for j in range(i + 2, n_nodes - 1):
                if i == 0 and j == n_nodes - 2:
                    continue
                total_checks += 1
                if intersect(tour[i], tour[i + 1], tour[j], tour[j + 1]):
                    total_intersections += 1

    pct_intersections = 0.0 if total_checks == 0 else total_intersections / total_checks


    return total_destinations_visited, total_time, per_truck_ok