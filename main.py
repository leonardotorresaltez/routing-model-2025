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
from core.utils.evaluation_utils import evaluate_solution
from core.utils.visualization_utils_plotly import create_routing_graph, visualize_routing_solution

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
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name, 
            name=cfg.run_name, 
            config=vars(cfg)
        )

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
    
    truck_starts = [truck.depot_idx for truck in data["trucks"]]
    print(f"len truck_starts:\n{len(truck_starts)}")
    
    print("TRUCK STARTS:", truck_starts)
    #sys.exit(0) 

    env = TSPEnv(
        nodes=nodes,
        source_mask=source_mask,
        initial_truck_positions=truck_starts,
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
        episode_reward = 0.0

        while not done:
            truck_id = env.active_truck
            action = agent.act(obs, truck_id)
            obs, reward, done, _, _ = env.step(action)

            agent.store_reward(reward)
            episode_reward += reward
            
        # Check constraints and compute reward inputs   
        total_destinations_visited, total_time = evaluate_solution(env, data, truck_starts, cfg)                

        for customer in data["customers"]:
            if customer.idx in [node for tour in env.tours for node in tour]:
                customer.delivered = True
            else:
                customer.delivered = False

        loss = agent.update()

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Total reward: {episode_reward:.3f} | "
                f"Loss: {loss:.4f}"
            )
            print("Tours shape: ", [len(tour) for tour in env.tours])
            print("Tours: ", env.tours)
            print("Total time: ", total_time)
            print("Total destinations visited: ", total_destinations_visited)
 
            G = create_routing_graph(data["depots"], data["customers"], env.tours, truck_starts)
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
    #path = f"checkpoints/{cfg.run_name}.pt"
    #torch.save(agent.policy.state_dict(), path)
    #print(f"--> SAVED: {path}")
    
#    if cfg.wandb:
#        wandb.finish()

if __name__ == "__main__":
    train()