import torch
import random
import numpy as np
import os
import wandb
from tqdm import tqdm

from configs.config import parse_args
from core.envs.tsp_env import TSPEnv
from core.models.agent import REINFORCEAgent

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    cfg = parse_args()
    set_seed(cfg.seed)
    
    #os.makedirs("checkpoints", exist_ok=True)
    
    # --- W&B Init ---
#    if cfg.wandb:
#        wandb.init(
#            project=cfg.project_name, 
#            name=cfg.run_name, 
#            config=vars(cfg)
#        )

    print(f"--> STARTING RUN: {cfg.run_name}")
    
    num_nodes = 50
    num_sources = 20

    nodes = torch.rand(num_nodes, 2)

    source_mask = np.zeros(num_nodes, dtype=bool)
    source_mask[:num_sources] = True

    initial_truck_positions = [0, 1, 0]   # both are sources

    env = TSPEnv(
        nodes=nodes,
        source_mask=source_mask,
        initial_truck_positions=initial_truck_positions
    )

    agent = REINFORCEAgent(cfg)

    # Training Loop
    # Using tqdm for a nice progress bar
    pbar = tqdm(range(cfg.episodes))
    print("episode is=", cfg.episodes)
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            truck_id = env.active_truck
            action = agent.act(obs, truck_id)
            obs, reward, done, _, _ = env.step(action)

            agent.store_reward(reward)
            total_reward += reward

        loss = agent.update()

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Total reward: {total_reward:.3f} | "
                f"Loss: {loss:.4f}"
            )

        # Logging to W&B
#        if cfg.wandb:
#            wandb.log({
#                "reward": episode_reward,
#                "loss": loss,
#                "episode": episode
#            })
            
        pbar.set_description(f"Rw: {total_reward:.2f}")

    # Save
    #path = f"checkpoints/{cfg.run_name}.pt"
    #torch.save(agent.policy.state_dict(), path)
    #print(f"--> SAVED: {path}")
    
#    if cfg.wandb:
#        wandb.finish()

if __name__ == "__main__":
    train()