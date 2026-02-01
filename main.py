import os
import random

import numpy as np
import torch
from tqdm import tqdm

import wandb
from configs.config import parse_args
from core.envs.tsp_env import TSPEnv
from core.models.agent import REINFORCEAgent
from core.utils.data_loader import MDVRPDataLoader
from core.utils.evaluation_utils import evaluate_solution
from core.utils.visualization_utils_plotly import create_routing_graph, visualize_routing_solution


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs("checkpoints", exist_ok=True)
    
    loader = MDVRPDataLoader(data_dir=cfg.data_dir)
    data = loader.load_data()

    # # Update config based on loaded data
    cfg.num_nodes = data["num_nodes"]
    
    # --- W&B Init ---
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name, 
            name=cfg.run_name, 
            config=vars(cfg)
        )

    print(f"--> STARTING RUN: {cfg.run_name}")
    
    # Node coordinates
        #example if self.nodes = 5
        #tensor([
        #[0.12, 0.77],   # node 0 (source)
        #[0.44, 0.91],   # node 1 (source)
        #[0.80, 0.13],   # node 2 (target)
        #[0.33, 0.59],   # node 3 (target)
        #[0.95, 0.22],   # node 4 (target)
        #])        
        
        
    # node_features: normalized time proximity profiles [N, N]
    nodes = data["node_features"] 
    print(f"Node coordinates:\n{nodes}") 
    print("Node shape:", nodes.shape)
    truck_starts = [truck.depot_idx for truck in data["trucks"]]
    print(f"Truck start positions (indices): {truck_starts}")

    customers = data["customers"]
    depots = data["depots"]

    # Initialize environment with multiple start positions
    env = TSPEnv(cfg, nodes, truck_starts, data["time_matrix"])

    agent = REINFORCEAgent(cfg, node_dim=cfg.num_nodes)

    # Training Loop
    # Using tqdm for a nice progress bar
    pbar = tqdm(range(cfg.episodes))
    for episode in pbar:
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            for truck_id in range(len(truck_starts)):
                if done: break
                action = agent.act(state, truck_id=truck_id)
                state, reward, done, _, _ = env.step(action, truck_id=truck_id)
                agent.store_reward(reward)
                episode_reward += reward.item()

        # Check constraints and compute reward inputs   
        total_destinations_visited, total_time = evaluate_solution(env, data, truck_starts, cfg)      
        
        for customer in customers:
            if customer.idx in [node for tour in env.tours for node in tour]:
                customer.delivered = True
            else:
                customer.delivered = False
        

        loss = agent.update()
        
        # Logging to console
        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Total reward: {episode_reward:.3f}| "
                f"Loss: {loss:.4f}"
            )
            print("Tours shape: ", [len(tour) for tour in env.tours])
            print("Tours: ", env.tours)
            print("Total time: ", total_time)
            print("Total destinations visited: ", total_destinations_visited)

            G = create_routing_graph(depots, customers, env.tours, truck_starts)
            visualize_routing_solution(G, step=episode, title_suffix="Final step", save_path=f"checkpoints/visualization_episode{episode}.html")
        

        # Logging to W&B
        if cfg.wandb:
            wandb.log({
                "reward": episode_reward,
                "loss": loss,
                "episode": episode
            })
            
        pbar.set_description(f"Rw: {episode_reward:.2f}")

    # Save
    path = f"checkpoints/{cfg.run_name}.pt"
    torch.save(agent.policy.state_dict(), path)
    print(f"--> SAVED: {path}")
    
    if cfg.wandb:
        wandb.finish()

if __name__ == "__main__":
    train()