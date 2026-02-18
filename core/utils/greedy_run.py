import torch

def run_greedy_episode(env, policy, edge_index, cfg, device):
    # 1. Unpack Gymnasium reset (state, info)
    state, _ = env.reset()
    
    terminated = False
    truncated = False
    current_truck = 0
    
    # Pre-calculate depot locations for masking
    cluster_ids = env.data["cluster_ids"].squeeze(1).to(device)
    is_depot = torch.zeros(env.num_nodes, dtype=torch.bool, device=device)
    for d in env.truck_starts:
        is_depot[d] = True

    # Move edge_index to device once
    edge_index = edge_index.to(device)

    # loop until either terminated or truncated
    while not (terminated or truncated):
        # 2. Ensure state is a tensor and on the right device
        if not isinstance(state, torch.Tensor):
            state_t = torch.FloatTensor(state).to(device)
        else:
            state_t = state.to(device)

        with torch.no_grad():
            # Match the policy forward signature (x, edge_index)
            _, node_logits, _ = policy(state_t, edge_index)

        # 3. Handle masking logic
        truck_mask, node_mask = env.mask_actions()
        node_mask = node_mask.to(device)

        # Find next valid truck that isn't masked
        tries = 0
        while truck_mask[current_truck] and tries < env.num_trucks:
            current_truck = (current_truck + 1) % env.num_trucks
            tries += 1
        
        if tries == env.num_trucks: 
            break

        # Apply Node and Cluster Masks
        node_logits[node_mask] = -1e9
        allowed_cluster = current_truck 
        
        wrong_cluster = (cluster_ids != allowed_cluster)
        node_logits[wrong_cluster & ~is_depot] = -1e9

        # Selection logic
        if torch.all(node_logits <= -1e8):
            node_action = env.truck_starts[current_truck]
        else:
            node_action = int(torch.argmax(node_logits).item())

        # Gym unpacking
        state, reward, terminated, truncated, info = env.step((current_truck, node_action))
        
        if "error" in info: 
            break 
        
        # Next truck for the next decision
        current_truck = (current_truck + 1) % env.num_trucks

    return env