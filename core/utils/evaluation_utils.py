from copy import deepcopy
import torch
from collections import Counter

def evaluate_solution(env, data, truck_starts, cfg):
    time_matrix = data["time_matrix"]
    num_trucks = len(env.tours)

    depot_set = set(env.depot_indices.tolist())
    full_tours = deepcopy(env.tours)

    # 1. Ensure each truck returns to depot
    for i in range(num_trucks):
        if full_tours[i][-1] != truck_starts[i]:
            full_tours[i].append(truck_starts[i])

    total_times = []
    per_truck_ok = []
    all_actual_customers_visited = []

    # 2. Compute time + visited customers
    for truck_id, tour in enumerate(full_tours):

        if tour[0] != truck_starts[truck_id]:
            raise ValueError(
                f"Truck {truck_id} started at {tour[0]}, not {truck_starts[truck_id]}"
            )

        total_time_truck = 0.0

        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            total_time_truck += float(time_matrix[u, v].item())

            if v not in depot_set:
                all_actual_customers_visited.append(v)

        total_times.append(total_time_truck)
        per_truck_ok.append(total_time_truck <= cfg.max_daily_delivery_time_each_truck)

    # 3. Duplicate customer detection
    if len(set(all_actual_customers_visited)) != len(all_actual_customers_visited):
        counts = Counter(all_actual_customers_visited)
        dupes = [node for node, count in counts.items() if count > 1]
        print(f"⚠️ Collision detected! Duplicate customers: {dupes}")
        #raise ValueError(f"Conflict: {len(dupes)} customers were visited by multiple trucks!")

    total_destinations_visited = len(set(all_actual_customers_visited))
    total_time = sum(total_times)

    # 4. Build node coordinate list
    nodes_list = data.get("nodes", [None] * data["num_nodes"])
    if nodes_list[0] is None:
        for d in data.get("depots", []):
            nodes_list[d.idx] = d
        for c in data.get("customers", []):
            nodes_list[c.idx] = c

    full_tours_coords = [
        [(nodes_list[node_idx].lat, nodes_list[node_idx].lon) for node_idx in tour]
        for tour in full_tours
    ]

    # 5. Intersection check (optional)
    def intersect(A, B, C, D):
        def ccw(P1, P2, P3):
            val = (P2[1] - P1[1]) * (P3[0] - P2[0]) - (P2[0] - P1[0]) * (P3[1] - P2[1])
            if abs(val) < 1e-9:
                return 0
            return 1 if val > 0 else -1

        return (ccw(A, B, C) != ccw(A, B, D) and ccw(C, D, A) != ccw(C, D, B))

    total_checks = 0
    total_intersections = 0

    for tour in full_tours_coords:
        n_points = len(tour)
        for i in range(n_points - 1):
            for j in range(i + 2, n_points - 1):
                if i == 0 and j == n_points - 2:
                    continue
                total_checks += 1
                if intersect(tour[i], tour[i + 1], tour[j], tour[j + 1]):
                    total_intersections += 1

    return total_destinations_visited, total_time, all(per_truck_ok)