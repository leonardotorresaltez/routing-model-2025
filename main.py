import os
import torch
import wandb
import numpy as np
import random 

# Optional: control BLAS threads
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "4"

from core.utils.data_loader import MDVRPDataLoader
from core.envs.gym_env import MDVRPGymEnv
from core.models.gnn_policy import GNNPolicy
from core.training.ppo import PPOAgent
from core.training.trainer import Trainer
from core.utils.greedy_run import run_greedy_episode
from core.utils.evaluation_utils import evaluate_solution
from core.utils.graph_utils import build_edge_index
from core.utils.visualization_utils import create_routing_graph, visualize_routing_solution
from configs.config import parse_args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    
    absolute_data_path = os.path.abspath(os.path.join(os.getcwd(), "data", cfg.data_dir))
    
    print(f"Loading data from: {cfg.data_dir}")
    loader = MDVRPDataLoader(data_dir=absolute_data_path)
    data = loader.load_data()
    print(f"Data loaded successfully. Nodes: {data['num_nodes']}")

    # -----------------------------
    # Environment Setup
    # -----------------------------
    # The new loader provides a consolidated 'nodes' list and 'time_matrix'
    env = MDVRPGymEnv(
        data,
        max_steps=cfg.max_steps if hasattr(cfg, 'max_steps') else 200,
        max_daily_time=cfg.max_daily_delivery_time_each_truck,
    )

    # Extract dimensions for the Neural Network
    # Note: env.observation_space usually represents [num_nodes, features_per_node]
    num_nodes = data["num_nodes"]
    num_trucks = len(data["trucks"])
    input_dim = data["node_features"].shape[1] 

    # -----------------------------
    # WandB Initialization
    # -----------------------------
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=vars(cfg),
        )

    # -----------------------------
    # Build Graph Structure (KNN)
    # -----------------------------
    # CHANGE: Your new loader's 'node_features' is a time-proximity profile.
    # If build_edge_index specifically requires lat/lon, you must extract them 
    # from the Depot/Customer objects since they aren't in the feature tensor anymore.
    coords = torch.tensor([n.location() for n in data["nodes"]], dtype=torch.float32).to(cfg.device)
    edge_index = build_edge_index(coords, k=10, device=cfg.device)

    # -----------------------------
    # PPO Agent & Policy
    # -----------------------------
    ppo_agent = PPOAgent(
        policy_class=GNNPolicy,
        input_dim=input_dim,
        hidden_dim=cfg.embed_dim,
        num_trucks=num_trucks,
        num_nodes=num_nodes,
        lr=cfg.lr,
        device=cfg.device,
    )

    print(f"Started trainer with {cfg.episodes} episodes.")
    trainer = Trainer(env, ppo_agent, edge_index, cfg)

    # -----------------------------
    # Training Loop
    # -----------------------------
    for episode in range(cfg.episodes):
        # 1. Training Phase
        batch = trainer.collect_rollout()
        stats = ppo_agent.update(batch)
        ep_return = batch["rewards"].sum().item()

        if cfg.wandb:
            wandb.log({
                "Train/Reward": ep_return,
                "Train/Loss": stats["loss"],
                "Train/Entropy": stats.get("entropy", 0.0),
            }, step=episode)

        # 2. Periodic Evaluation
        if episode % 50 == 0:
            print(f"\n[Episode {episode}] Performing Greedy Evaluation...")
            greedy_env = run_greedy_episode(env, ppo_agent.policy, edge_index, cfg, cfg.device)

            # Extract starting positions for evaluation
            truck_starts = [t.depot_idx for t in data["trucks"]]
            
            visited, total_time, per_truck_ok = evaluate_solution(
                greedy_env,
                data,
                truck_starts=truck_starts,
                cfg=cfg,
            )

            print(f"Eval -> Visited: {visited}/{num_nodes - len(data['depots'])} | Time: {total_time:.2f} | Rewards:{ep_return:.2f} | Loss:{stats["loss"]}")

            if cfg.wandb:
                wandb.log({
                    "Eval/Total_reward": ep_return,
                    "Eval/Total_time": total_time,
                    "Eval/Visited_nodes": visited,
                }, step=episode)

            # 3. Visualization
            #print(f"Generating Routing Plot...")
           
            G = create_routing_graph(depots=data["depots"],customers=data["customers"],tours=greedy_env.tours,truck_starts=truck_starts,node_to_idx=data["node_to_idx"],idx_to_node=data["idx_to_node"],time_matrix=data["time_matrix"].cpu().numpy(),)

            visualize_routing_solution(G,step=episode,title_suffix=f"Ep {episode}",save_path=f"checkpoints/vis_ep{episode}.png",)

    # -----------------------------
    # Final Wrap-up
    # -----------------------------
    print("\nTraining complete.")
    if cfg.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()