import os
import sys

import torch
from tqdm import tqdm

import wandb
from configs.config import parse_args
from core.envs.tsp_env import MDVRPEnv
from core.models.agent import MDVRPREINFORCEAgent
from core.utils.data_loader import MDVRPDataLoader


def train():
    cfg = parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load Real Data
    loader = MDVRPDataLoader()
    data = loader.load_data()

    if cfg.wandb:
        wandb.init(project="mdvrp-rl", name=cfg.run_name, config=vars(cfg))

    env = MDVRPEnv(cfg, data)
    agent = MDVRPREINFORCEAgent(cfg, data)

    pbar = tqdm(range(cfg.episodes))
    for episode in pbar:
        state, _ = env.reset()
        
        # One-shot action
        action = agent.act(state)
        state, reward, _, _, info = env.step(action)
        
        agent.store_reward(reward)
        loss = agent.update()
        
        if cfg.wandb:
            wandb.log({"reward": reward, "loss": loss, "total_time": info["total_time"]})
            
        pbar.set_description(f"Time: {info['total_time']:.2f}h")

    torch.save(agent.policy.state_dict(), f"checkpoints/mdvrp_{cfg.run_name}.pt")
    if cfg.wandb: wandb.finish()

if __name__ == "__main__":
    train()
