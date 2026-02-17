import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch

from core.utils.data_loader import MDVRPDataLoader
from core.envs.gym_env import MDVRPGymEnv
from core.models.gnn_policy import GNNPolicy
from core.training.ppo import PPOAgent
from core.training.trainer import Trainer
from core.utils.greedy_run import run_greedy_episode
from core.utils.evaluation_utils import evaluate_solution
from core.utils.graph_utils import build_edge_index
from core.utils.visualization_utils import create_routing_graph, visualize_routing_solution

import wandb


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 128
    lr = 3e-4
    gamma = 0.99
    eps_clip = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    episodes = 300
    max_daily_delivery_time_each_truck = 24.0

    # wandb-related
    wandb = True          # set to False to disable logging
    project_name = "mdvrp-gnn-ppo"
    run_name = "cluster_aware_ppo_run_1"


def main():
    cfg = Config()
    #data with clustering
    print("K-Means is processing")
    loader = MDVRPDataLoader()
    data = loader.load_data()
    print("Data loaded succesfully")
    num_nodes = data["num_nodes"]
    num_trucks = len(data["trucks"])
    input_dim = data["node_features"].shape[1]   # 5: lat, lon, demand, visited, cluster_id

    # init wandb
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=vars(cfg),
        )

   
    edge_index = build_edge_index(num_nodes).to(cfg.device)
    env = MDVRPGymEnv(data, max_steps=1000, max_daily_time=24.0)
    policy = GNNPolicy(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_trucks=num_trucks,
        num_nodes=num_nodes,
    ).to(cfg.device)

    ppo_agent = PPOAgent(
        policy=policy,
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_trucks=num_trucks,
        num_nodes=num_nodes,
        lr=cfg.lr,
        gamma=cfg.gamma,
        eps_clip=cfg.eps_clip,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        device=cfg.device,
    )

    print("Started trainer")
    trainer = Trainer(env, policy, ppo_agent, edge_index, cfg)
    for episode in range(cfg.episodes):
        
        batch = trainer.collect_rollout()
        stats = ppo_agent.update(batch)
        ep_return = batch["rewards"].sum().item()

        if episode % 50 == 0:
            print(
                f"\nEpisode {episode:4d} | "
                f"Total reward: {ep_return:.3f} | "
                f"Loss: {stats['loss']:.4f}"
            )

            # Run greedy rollout for visualization
            greedy_env = run_greedy_episode(env, policy, edge_index, cfg, cfg.device)

            # Evaluate solution
            visited, total_time, per_truck_ok = evaluate_solution(
                greedy_env,
                data,
                truck_starts=[t.depot_idx for t in data["trucks"]],
                cfg=cfg,
            )

            print(f"Total time: {total_time:.2f}")
            print(f"Total destinations visited: {visited}")
            print(f"Per-truck time OK: {per_truck_ok}")

            delivered_nodes = {node for tour in greedy_env.tours for node in tour}
            for customer in data["customers"]:
                customer.delivered = customer.idx in delivered_nodes

            if cfg.wandb:
                wandb.log(
                    {
                        "Total reward": ep_return,
                        "Last Loss": stats["loss"],
                        "Episode": episode,
                        "Total time": total_time,
                        "Total destinations visited": visited,
                    }
                )
            G = create_routing_graph(
                data["depots"],
                data["customers"],
                greedy_env.tours,
                truck_starts=[t.depot_idx for t in data["trucks"]],
                node_to_idx=data["node_to_idx"],
                idx_to_node=data["idx_to_node"],
                time_matrix=data["time_matrix"]
            )

            visualize_routing_solution(
                G,
                step=episode,
                title_suffix="Training Snapshot",
                save_path=f"checkpoints/visualization_episode{episode}.png",
            )

   
    # Greedy evalution
    greedy_env = run_greedy_episode(env, policy, edge_index, cfg, cfg.device)

    visited, total_time, per_truck_ok = evaluate_solution(
        greedy_env,
        data,
        truck_starts=[t.depot_idx for t in data["trucks"]],
        cfg=cfg,
    )

    print(f"Total customers visited: {visited}")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Per-truck time OK: {per_truck_ok}")

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()