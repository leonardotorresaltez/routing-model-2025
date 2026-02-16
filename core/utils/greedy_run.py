import torch

def run_greedy_episode(env, policy, edge_index, cfg, device):
    state = env.reset()
    done = False
    current_truck = 0

    # Truck → cluster mapping 
    truck_cluster = {0: 0, 1: 1}
    cluster_ids = env.data["cluster_ids"].squeeze(1)  # (num_nodes,)

    while not done:
        state_t = state.to(device).unsqueeze(0)
        N = state_t.size(1)
        flat_state = state_t.view(N, -1)

        with torch.no_grad():
            _, node_logits, _ = policy(flat_state, edge_index)

        # Normal masking
        truck_mask, node_mask = env.mask_actions()

        # Force a valid truck
        tries = 0
        while truck_mask[current_truck] and tries < env.num_trucks:
            current_truck = (current_truck + 1) % env.num_trucks
            tries += 1

        # If all trucks are masked, stop
        if tries == env.num_trucks and truck_mask[current_truck]:
            break

        node_logits[node_mask] = -1e9

        # Cluster masking
        allowed_cluster = truck_cluster[current_truck]
        cluster_mask = (cluster_ids != allowed_cluster)
        node_logits[cluster_mask] = -1e9

        # If all nodes are masked, stop
        if torch.all(node_logits == -1e9):
            break

        # Greedy node choice
        node_action = int(torch.argmax(node_logits).item())

        # Step environment
        state, reward, done, _ = env.step((current_truck, node_action))

        # Enforce daily time limit
        if env.truck_times[current_truck] > cfg.max_daily_delivery_time_each_truck:
            print(f"Truck {current_truck} exceeded daily limit ({env.truck_times[current_truck]:.2f}h).")
            done = True

        # Next truck
        current_truck = (current_truck + 1) % env.num_trucks

    return env