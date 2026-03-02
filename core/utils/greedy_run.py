import torch
def run_greedy_episode(env, policy, edge_index, cfg, device):
    state, _ = env.reset()
    done = False
    
    while not done:
        # 1. Get the current mask (Nodes visited in PREVIOUS steps)
        t_mask, base_n_mask = env.mask_actions() 
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            _, logits, _ = policy(state_t, edge_index) # Shape: [1, num_trucks, num_nodes]

        # 2. COORDINATE: Make trucks pick one-by-one in this millisecond
        actions = []
        current_turn_mask = base_n_mask.clone().to(device)

        for t in range(env.num_trucks):
            # Apply the mask (base mask + nodes already picked by other trucks this turn)
            truck_logits = logits[0, t].masked_fill(current_turn_mask[t], -1e10)
            
            # Pick the best available node
            action = torch.argmax(truck_logits).item()
            actions.append(action)

            # If this truck picked a customer, hide it from all other trucks immediately
            if action not in env.depot_indices.tolist():
                current_turn_mask[:, action] = True 

        # 3. Send the coordinated actions to your environment
        state, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        
    return env