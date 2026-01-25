import torch
import random
import numpy as np
import os
import wandb
from tqdm import tqdm

from configs.config import parse_args
from core.envs.tsp_env import TSPEnv
from core.models.agent import REINFORCEFleetAgent

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs("checkpoints", exist_ok=True)
    
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
    nodes = torch.rand(cfg.num_sources + cfg.num_targets, 2)

    env = TSPEnv(cfg, nodes)
    agent = REINFORCEFleetAgent(cfg)

    # Training Loop
    # Using tqdm for a nice progress bar
    pbar = tqdm(range(cfg.episodes))
    print("episode is=", cfg.episodes)
    for episode in pbar:
        state, _ = env.reset()
        terminated = False
        episode_reward = 0.0
        
        while not terminated:
            action = agent.act(state)
            state, reward, terminated = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward.item()
            
        loss = agent.update()
        
        # Logging to console
        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Total reward: {episode_reward:.3f}"
            )

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