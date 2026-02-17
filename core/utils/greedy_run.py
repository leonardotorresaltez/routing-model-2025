import torch
def run_greedy_episode(env, policy, edge_index, cfg, device):
    state = env.reset()
    done = False
    current_truck = 0
    truck_cluster = {i: i for i in range(env.num_trucks)}
    cluster_ids = env.data["cluster_ids"].squeeze(1)

    while not done:
        state_t = state.to(device).unsqueeze(0)
        with torch.no_grad():
            _, node_logits, _ = policy(state_t.view(state_t.size(1), -1), edge_index)

        truck_mask, node_mask = env.mask_actions()

        # Find next valid truck
        tries = 0
        while truck_mask[current_truck] and tries < env.num_trucks:
            current_truck = (current_truck + 1) % env.num_trucks
            tries += 1
        if tries == env.num_trucks: break

        # Apply Masks
        node_logits[node_mask] = -1e9
        allowed_cluster = truck_cluster.get(current_truck)
        
        is_depot = torch.zeros_like(cluster_ids, dtype=torch.bool)
        for d in env.truck_starts: is_depot[d] = True
        
        wrong_cluster = (cluster_ids != allowed_cluster)
        node_logits[wrong_cluster & ~is_depot] = -1e9

        if torch.all(node_logits == -1e9):
            # If blocked, force return to depot to avoid crash
            node_action = env.truck_starts[current_truck]
        else:
            node_action = int(torch.argmax(node_logits).item())

        state, reward, done, info = env.step((current_truck, node_action))
        
        if "error" in info: break 
        
        current_truck = (current_truck + 1) % env.num_trucks
    return env