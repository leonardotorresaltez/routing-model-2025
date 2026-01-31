import torch
import random
import numpy as np
import os
import wandb
from tqdm import tqdm
import sys

from configs.config import parse_args
from core.envs.tsp_env import TSPEnv
from core.models.agent import REINFORCEAgent
from core.utils.data_loader import MDVRPDataLoader

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    cfg = parse_args()
    set_seed(cfg.seed)
    
    #os.makedirs("checkpoints", exist_ok=True)
    
    loader = MDVRPDataLoader(data_dir=cfg.data_dir)
    data = loader.load_data()
    
    # # Update config based on loaded data
    #TODO 
    cfg.num_nodes = data["num_nodes"]   
    
    # --- W&B Init ---
#    if cfg.wandb:
#        wandb.init(
#            project=cfg.project_name, 
#            name=cfg.run_name, 
#            config=vars(cfg)
#        )

    print(f"--> STARTING RUN: {cfg.run_name}")
    
    #num_nodes = 50
    num_nodes = cfg.num_nodes
    #num_sources = 20

    #nodes = torch.rand(num_nodes, 2)

    #source_mask = np.zeros(num_nodes, dtype=bool)
    #source_mask[:num_sources] = True
    
    # Determinar source_mask usando la propiedad isSource de cada nodo
    #source_mask = np.array([getattr(node, 'isSource', False) for node in data["nodes"]], dtype=bool)    

    #initial_truck_positions = [0, 1, 0]   # both are sources
    
    
    
    # newwwwww
    nodesObjs = data["nodes"] 
    nodes = torch.tensor([[n.lat, n.lon] for n in nodesObjs], dtype=torch.float32)

    print(f"num_nodes:{num_nodes}") 
    print(f"nodesObjs  size:{len(nodesObjs)}") 
    
    source_mask = np.array([getattr(node, 'isSource', False) for node in nodesObjs], dtype=bool)  
    print(type(source_mask))
    #print(f"source_mask:\n{source_mask}")
    #print(f"Nodes :\n{nodes}") 
    
    print(f"number of depots:\n{len(data['depots'])}")  
    print(f"number of customers:\n{len(data['customers'])}")  
    
    print(f"number of trucks:\n{len(data['trucks'])}")
    
    initial_truck_positions = [truck.depot_idx for truck in data["trucks"]]
    print(f"len truck_starts:\n{len(initial_truck_positions)}")
    
    print("TRUCK STARTS:", initial_truck_positions)
    #sys.exit(0) 

    env = TSPEnv(
        nodes=nodes,
        source_mask=source_mask,
        initial_truck_positions=initial_truck_positions,
        time_matrix=data["time_matrix"]      
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