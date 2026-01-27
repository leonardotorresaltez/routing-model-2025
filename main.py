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
    """
    Reset: Start a fresh day with all customers unvisited.
    Act: The Agent builds a full plan for the day.
    Step: The Environment calculates how long that plan took.
    Learn: Every 10 episodes, the Agent looks back at what worked and updates the Policy's weights via loss.backward().
    Repeat: This continues for hundreds of episodes until the Agent learns the spatial patterns of the customers.
    """
    cfg = parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load Real Data
    loader = MDVRPDataLoader(data_dir=cfg.data_dir)
    data = loader.load_data()

    if cfg.wandb:
        wandb.init(project="mdvrp-rl", name=cfg.run_name, config=vars(cfg))

    env = MDVRPEnv(cfg, data)
    agent = MDVRPREINFORCEAgent(cfg, data)
    batch_rewards = []

    print(f"--> STARTING RUN: {cfg.run_name}")
    pbar = tqdm(range(cfg.episodes))
    for episode in pbar:
        state, _ = env.reset()
        
        # One-shot action
        # print('state ', state)
        action = agent.act(state)
        state, reward, _, _, info = env.step(action)
        
        agent.store_reward(reward)
        batch_rewards.append(reward) # Track rewards for this batch
    
        if (episode + 1) % 10 == 0:
            
            for truck_id, route in action.items():
                if route: # Only print trucks that actually moved
                    print(f"Truck {truck_id}: {route}")
            
            
            loss = agent.update()
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            pbar.write(
                    f"Episode {episode+1:>4} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Loss: {loss:.4f} | "
                    f"Time: {info['total_time']:.2f}h"
                )
            
            if cfg.wandb:
                wandb.log({
                    "episode ":episode,
                    "batch_avg_reward ": avg_reward, 
                    "loss ": loss, 
                    "total_time ": info["total_time"]
                })
            batch_rewards = [] # Reset for next batch
            

    torch.save(agent.policy.state_dict(), f"checkpoints/mdvrp_{cfg.run_name}.pt")
    if cfg.wandb: wandb.finish()

if __name__ == "__main__":
    train()
