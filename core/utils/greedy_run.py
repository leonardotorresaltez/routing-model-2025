import torch

def run_greedy_episode(env, policy, edge_index, cfg, device):
    state, _ = env.reset()
    edge_index = edge_index.to(device)
    depot_indices = env.depot_indices.to(device)
    
    for _ in range(env.max_steps):
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            _, n_logits, _ = policy(state_t, edge_index)
        
        n_logits = n_logits.squeeze(0) 
        _, n_mask = env.mask_actions()
        n_mask = n_mask.to(device)
        n_logits[n_mask] = -1e9

        joint_action = []
        step_taken_mask = torch.zeros(env.num_nodes, dtype=torch.bool, device=device)

        for t_id in range(env.num_trucks):
            truck_logits = n_logits[t_id].clone()
            is_customer = torch.ones(env.num_nodes, dtype=torch.bool, device=device)
            for d_idx in depot_indices:
                is_customer[d_idx] = False
            truck_logits[step_taken_mask & is_customer] = -1e9
            
            best_node = torch.argmax(truck_logits).item()
            joint_action.append(best_node)
        
            if best_node not in depot_indices:
                step_taken_mask[best_node] = True
        
        state, _, done, trunc, _ = env.step(joint_action)
        
        if done or trunc: 
            break
            
    return env