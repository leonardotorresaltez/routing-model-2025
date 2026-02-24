import torch
from torch.distributions import Categorical

class Trainer:
    def __init__(self, env, ppo_agent, edge_index, cfg):
        self.env = env
        self.ppo = ppo_agent 
        self.edge_index = edge_index.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device

    def collect_rollout(self):
        data = {
            "states": [], "node_actions": [], 
            "rewards": [], "masks": [], "log_probs": []
        }
        state, info = self.env.reset()
        depot_indices = self.env.depot_indices.to(self.device)

        for step in range(self.env.max_steps):
            state_tensor = torch.FloatTensor(state).to(self.device) 
            with torch.no_grad():
                _, node_logits, _ = self.ppo.policy(state_tensor.unsqueeze(0), self.edge_index)
            
            node_logits = node_logits.squeeze(0) # [num_trucks, num_nodes]
            _, n_mask = self.env.mask_actions()
            n_mask = n_mask.to(self.device)
            joint_action = torch.zeros(self.env.num_trucks, dtype=torch.long, device=self.device)
            joint_log_probs = torch.zeros(self.env.num_trucks, device=self.device)
            tick_visited_mask = torch.zeros(self.env.num_nodes, dtype=torch.bool, device=self.device)

            for t_id in range(self.env.num_trucks):
                t_logits = node_logits[t_id].clone()
                t_logits[n_mask[t_id]] = -1e9
                customer_mask = torch.ones(self.env.num_nodes, dtype=torch.bool, device=self.device)
                customer_mask[depot_indices] = False
                t_logits[tick_visited_mask & customer_mask] = -1e9
                dist = Categorical(logits=t_logits)
                action = dist.sample()
                
                joint_action[t_id] = action
                joint_log_probs[t_id] = dist.log_prob(action)
                if action not in depot_indices:
                    tick_visited_mask[action] = True

            next_state, reward, terminated, truncated, info = self.env.step(joint_action.cpu().numpy())
            
            data["states"].append(state_tensor.cpu())
            data["node_actions"].append(joint_action.cpu()) 
            data["rewards"].append(torch.tensor(reward, dtype=torch.float32))
            data["log_probs"].append(joint_log_probs.sum().cpu()) 
            data["masks"].append(torch.tensor(1.0 - float(terminated or truncated)))
            
            state = next_state
            if terminated or truncated: break

        return {
            "states": torch.stack(data["states"]), 
            "node_actions": torch.stack(data["node_actions"]), 
            "rewards": torch.stack(data["rewards"]),
            "masks": torch.stack(data["masks"]), 
            "log_probs_old": torch.stack(data["log_probs"]), 
            "edge_index": self.edge_index
        }

    def train(self):
        print(f"Starting training on {self.device}...")
        for episode in range(self.cfg.episodes):
            batch = self.collect_rollout()
            stats = self.ppo.update(batch)
            ep_return = batch["rewards"].sum().item()
            visited = self.env.visited_customers.sum().item()
            if episode % 1 == 0:
                print(
                    f"Ep {episode:03d} | "
                    f"Return: {ep_return:7.2f} | "
                    f"Visited: {int(visited):02d} | "
                    f"Loss: {stats['loss']:.4f} | "
                    f"Entropy: {stats.get('entropy', 0):.3f}"
                )