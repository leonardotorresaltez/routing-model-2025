def evaluate_solution(tours, data, truck_starts, cfg):
    # Check total time
    total_times = []
    for truck_id, tour in enumerate(tours):
        
        total_time_truck = 0.0
        for i in range(len(tour)-1):
            from_node = tour[i]
            to_node = tour[i+1]
            travel_time = data["time_matrix"][from_node, to_node]
            total_time_truck += travel_time
        # total_time_truck += data["time_matrix"][truck_starts[truck_id], tour[0]]   # From depot to first
        total_time_truck += data["time_matrix"][tour[-1], truck_starts[truck_id]]  # Return to depot
        
        #if total_time_truck > cfg.max_daily_delivery_time_each_truck:
        #    raise ValueError(f"Truck {truck_id} exceeded max daily delivery time: {total_time_truck} > {cfg.max_daily_delivery_time_each_truck}")

        if tour[0] != truck_starts[truck_id]:
            raise ValueError(f"Truck {truck_id} did not start at its depot: started at {tour[0]}, should start at {truck_starts[truck_id]}")

        total_times.append(total_time_truck)

    all_destinations_visited = [subtour for tour in tours for subtour in tour[1:]]
    if len(set(all_destinations_visited)) != len(all_destinations_visited):
        raise ValueError("Some destinations were visited more than once!")
    
    total_destinations_visited = len(all_destinations_visited)
    total_time = sum(total_times).item()

    return total_destinations_visited, total_time