import torch
from torch.distributions import Categorical

class Trainer:
    def __init__(self, env, policy, ppo_agent, edge_index, cfg):
        self.env = env
        self.policy = policy
        self.ppo = ppo_agent
        self.edge_index = edge_index.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device

    def collect_rollout(self):
        states = []
        truck_actions = []
        node_actions = []
        rewards = []
        dones = []
        log_probs = []

        state = self.env.reset()
        done = False

        for _ in range(self.env.max_steps):
            state_t = state.to(self.device).unsqueeze(0)  # (1, N, F)
            N = state_t.size(1)
            flat_state = state_t.view(N, -1)

            with torch.no_grad():
                truck_logits, node_logits, _ = self.policy(flat_state, self.edge_index)

            truck_mask, node_mask = self.env.mask_actions()
            truck_logits[truck_mask] = -1e9
            node_logits[node_mask] = -1e9
            
            truck_dist = Categorical(logits=truck_logits)
            node_dist = Categorical(logits=node_logits)

            truck_action = truck_dist.sample()
            node_action = node_dist.sample()

            truck_cluster = {0: 0, 1: 1}

            cluster_ids = self.env.data["cluster_ids"].squeeze(1)  # (num_nodes,)
            allowed_cluster = truck_cluster[truck_action.item()]

            cluster_mask = (cluster_ids != allowed_cluster)
            node_logits[cluster_mask] = -1e9

            # Recompute node distribution after masking
            node_dist = Categorical(logits=node_logits)
            node_action = node_dist.sample()

            log_prob = truck_dist.log_prob(truck_action) + node_dist.log_prob(node_action)

            next_state, reward, done, _ = self.env.step(
                (truck_action.item(), node_action.item())
            )

            states.append(state_t.squeeze(0).cpu())
            truck_actions.append(truck_action.cpu())
            node_actions.append(node_action.cpu())
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            dones.append(torch.tensor(float(done)))
            log_probs.append(log_prob.cpu())

            state = next_state
            if done:
                break

        return {
            "states": torch.stack(states),
            "truck_actions": torch.stack(truck_actions),
            "node_actions": torch.stack(node_actions),
            "rewards": torch.stack(rewards),
            "dones": torch.stack(dones),
            "log_probs": torch.stack(log_probs),
            "edge_index": self.edge_index,
        }

    def train(self):
        for episode in range(self.cfg.episodes):
            batch = self.collect_rollout()
            stats = self.ppo.update(batch)

            ep_return = batch["rewards"].sum().item()
            print(
                f"Episode {episode} | "
                f"Loss: {stats['loss']:.3f} | "
                f"Policy: {stats['policy_loss']:.3f} | "
                f"Value: {stats['value_loss']:.3f} | "
                f"Reward: {ep_return:.2f}"
            )