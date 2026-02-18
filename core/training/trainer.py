import torch
from torch.distributions import Categorical

class Trainer:
    def __init__(self, env, ppo_agent, edge_index, cfg):
        self.env = env
        self.ppo = ppo_agent # This now contains the policy
        self.edge_index = edge_index.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device

    def collect_rollout(self):
        # Arrays to store trajectory data
        states, truck_actions, node_actions, rewards, masks, log_probs = [], [], [], [], [], []
        
        # Gymnasium reset returns (obs, info)
        state, info = self.env.reset()
        
        # Configuration for cluster masking
        truck_cluster_map = {i: i for i in range(self.env.num_trucks)}
        cluster_ids = self.env.data["node_features"][:, 4].to(self.device) # Assuming index 4 is cluster
        
        # Pre-calculate depot locations for masking
        is_depot = torch.zeros(self.env.num_nodes, dtype=torch.bool, device=self.device)
        for start_node in self.env.truck_starts:
            is_depot[start_node] = True

        for step in range(self.env.max_steps):
            # 1. Prepare state for policy
            # state is currently a numpy array from Gym, convert to tensor
            state_tensor = torch.FloatTensor(state).to(self.device) 
            
            with torch.no_grad():
                # Policy head (truck_logits, node_logits, value)
                truck_logits, node_logits, _ = self.ppo.policy(state_tensor, self.edge_index)

           #Mask 
            truck_mask, node_mask = self.env.mask_actions()
            truck_mask = truck_mask.to(self.device)
            node_mask = node_mask.to(self.device)

            # Apply truck mask
            truck_logits[truck_mask] = -1e9
            truck_dist = Categorical(logits=truck_logits)
            truck_action = truck_dist.sample()

            # Apply node mask and cluster constraints
            node_logits[node_mask] = -1e9
            allowed_cluster = truck_cluster_map.get(truck_action.item())
            if allowed_cluster is not None:
                wrong_cluster = (cluster_ids != allowed_cluster)
                node_logits[wrong_cluster & ~is_depot] = -1e9

            # Safety: Force depot if trapped
            if torch.all(node_logits <= -1e8):
                node_logits[self.env.truck_starts[truck_action.item()]] = 0 
            
            node_dist = Categorical(logits=node_logits)
            node_action = node_dist.sample()

            # Env step
            action = (truck_action.item(), node_action.item())
            # Gymnasium step returns 5 values
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Log prob calc
            current_log_prob = truck_dist.log_prob(truck_action) + node_dist.log_prob(node_action)

            # storing
            states.append(state_tensor.cpu())
            truck_actions.append(truck_action.cpu())
            node_actions.append(node_action.cpu())
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            masks.append(torch.tensor(1.0 - float(terminated)))
            log_probs.append(current_log_prob.cpu())

            state = next_state
            if terminated or truncated:
                break

        return {
            "states": torch.stack(states), 
            "truck_actions": torch.stack(truck_actions), 
            "node_actions": torch.stack(node_actions), 
            "rewards": torch.stack(rewards),
            "masks": torch.stack(masks), 
            "log_probs_old": torch.stack(log_probs), 
            "edge_index": self.edge_index
        }

    def train(self):
        for episode in range(self.cfg.episodes):
            batch = self.collect_rollout()
            stats = self.ppo.update(batch)

            ep_return = batch["rewards"].sum().item()
            print(
                f"Episode {episode} | "
                f"Loss: {stats['loss']:.3f} | "
                f"Reward: {ep_return:.2f} | "
                f"Steps: {len(batch['rewards'])}"
            )