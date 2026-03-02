import torch
from torch.distributions import Categorical

class Trainer:
    def __init__(self, env, agent, edge_index, cfg, episodes_per_batch=8):
        self.env = env
        self.agent = agent
        self.edge_index = edge_index
        self.cfg = cfg
        self.episodes_per_batch = episodes_per_batch

    def collect_rollout(self):
        all_states, all_actions, all_log_probs = [], [], []
        all_rewards, all_masks, all_node_masks = [], [], []

        for _ in range(self.episodes_per_batch):
            state, _ = self.env.reset()
            done = False
            truncated = False

            while True:
                # 1. Get the base mask from env (already visited in previous steps)
                t_mask, base_n_mask = self.env.mask_actions()
                state_tensor = state.clone().detach().unsqueeze(0).to(self.agent.device)

                with torch.no_grad():
                    _, node_logits, _ = self.agent.policy(state_tensor, self.edge_index)

                # --- THE CRITICAL FIX: SEQUENTIAL SAMPLING ---
                # We need to pick actions one-by-one to prevent intra-step collisions
                actions_list = []
                log_probs_list = []
                # Working copy of the mask for this specific timestep
                current_n_mask = base_n_mask.clone().to(self.agent.device)

                for t in range(self.env.num_trucks):
                    # Mask out nodes already visited OR nodes picked by previous trucks this turn
                    truck_logits = node_logits[0, t].masked_fill(current_n_mask[t], -1e10)
                    
                    dist = Categorical(logits=truck_logits)
                    sampled_action = dist.sample()
                    
                    actions_list.append(sampled_action)
                    log_probs_list.append(dist.log_prob(sampled_action))

                    # If this truck picked a customer, mask it for all OTHER trucks in this turn
                    picked_node = sampled_action.item()
                    if picked_node not in self.env.depot_indices.tolist():
                        current_n_mask[:, picked_node] = True 

                # Combine back into tensors
                actions = torch.stack(actions_list).long() # [num_trucks]
                log_prob = torch.stack(log_probs_list).sum().detach() # Global log_prob for this state
                # ---------------------------------------------

                next_state, reward, done, truncated, _ = self.env.step(actions.cpu().numpy())

                # Store rollout data
                all_states.append(state.clone().detach())
                all_actions.append(actions.cpu())
                all_log_probs.append(log_prob.cpu())
                all_rewards.append(torch.tensor(reward, dtype=torch.float32))
                all_masks.append(torch.tensor(0.0 if (done or truncated) else 1.0))
                # Store the base_n_mask (what the model saw before taking actions)
                all_node_masks.append(base_n_mask.clone().detach().bool())

                state = next_state
                if done or truncated:
                    break

        batch = {
            "states": torch.stack(all_states),
            "node_actions": torch.stack(all_actions),
            "log_probs_old": torch.stack(all_log_probs),
            "rewards": torch.stack(all_rewards),
            "masks": torch.stack(all_masks),
            "node_masks": torch.stack(all_node_masks),
            "edge_index": self.edge_index,
        }
        return batch