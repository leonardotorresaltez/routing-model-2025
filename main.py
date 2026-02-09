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
    
    num_nodes = data["num_nodes"]   
    
    # --- Wandb Init ---
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name, 
            name=cfg.run_name, 
            config=vars(cfg)
        )

    print(f"--> STARTING RUN: {cfg.run_name}")

    nodesObjs = data["nodes"] 
    nodes = torch.tensor([[n.lat, n.lon] for n in nodesObjs], dtype=torch.float32)
    #print(nodes)   
    #sys.exit(0)
    # --- prints just for verification ---
    print(f"num_nodes:{num_nodes}") 
    print(f"nodesObjs  size:{len(nodesObjs)}") 
    print(f"number of depots:\n{len(data['depots'])}")  
    print(f"number of customers:\n{len(data['customers'])}")    
    print(f"number of trucks:\n{len(data['trucks'])}")    
    
    # Array, Create source mask for depots to avoid visiting them as targets
    source_mask = np.array([getattr(node, 'isSource', False) for node in nodesObjs], dtype=bool)  
   
    
    # List, Initial truck positions (at their depots)
    truck_starts = [truck.depot_idx for truck in data["trucks"]]
    print(f"len truck_starts:\n{len(truck_starts)}")
    
    print("TRUCK STARTS:", truck_starts)


    env = TSPEnv(
        cfg=cfg,
        nodes=nodes,
        source_mask=source_mask,
        truck_starts=truck_starts,
        time_matrix=data["time_matrix"]      
    )

    agent = REINFORCEAgent(cfg, data["time_matrix"])

    # Training Loop, tqdm for a nice progress bar
    pbar = tqdm(range(cfg.episodes))
    print("episode is=", cfg.episodes)
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        terminated = False
        episode_reward = 0.0
        reward = 0.0

        while not (done or terminated):
            truck_id = env.active_truck
            action = agent.act(obs, truck_id, env.trucks_dict_state)
            obs, reward, done, terminated, _ = env.step(action)
            

            agent.store_reward(reward)
            episode_reward += reward

        # Check constraints and compute reward inputs
        all_tours = [truck_state.tour for truck_state in env.trucks_dict_state.values()]   #TODO construir la lista dentro
        total_destinations_visited, total_time = evaluate_solution(all_tours, data, truck_starts, cfg)                

        for customer in data["customers"]:
            if customer.idx in [node for tour in all_tours for node in tour]:
                customer.delivered = True
            else:
                customer.delivered = False

        loss = agent.update()

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Total reward: {episode_reward:.3f} | "
                f"last Loss: {loss:.4f} | "
                f"last reward: {reward:.4f}" 
            )
            print("Total time: ", total_time)
            print("Total destinations visited: ", total_destinations_visited)
            pbar.write("\n--- Sample Route Plan ---")

            # total time and tour for each truck
            for i, truck_state in env.trucks_dict_state.items():
                print(f"Truck {i}: total time = {truck_state.total_time}, tour = {truck_state.tour}")
            pbar.write("-------------------------\n")
 
 
            # Visualization
            G = create_routing_graph(data["depots"], data["customers"], all_tours, truck_starts)
            visualize_routing_solution(G, step=episode, title_suffix="Final step", save_path=f"checkpoints/visualization_episode{episode}.html")
             
                        

        # Logging to W&B
        if cfg.wandb:
            wandb.log({
                "Total reward": episode_reward,
                "Last Loss": loss,
                "Episode": episode,
                "Total time": total_time,
                "Total destinations visited": total_destinations_visited
            })
            
        pbar.set_description(f"Rw: {episode_reward:.2f}")

    # Save
    #path = f"checkpoints/{cfg.run_name}.pt"
    #torch.save(agent.policy.state_dict(), path)
    #print(f"--> SAVED: {path}")
    
    if cfg.wandb:
        wandb.finish()

if __name__ == "__main__":
    train()