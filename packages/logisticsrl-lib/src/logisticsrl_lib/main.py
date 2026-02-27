import torch
import random
import numpy as np
import os
import wandb
from tqdm import tqdm
import sys

from logisticsrl_lib.configs.config import parse_args
from logisticsrl_lib.reinforcelearning.tsp_env import TSPEnv
from logisticsrl_lib.reinforcelearning.agent import REINFORCEAgent
from loader_lib.data_loader import FleetStatus, MDVRPDataLoader, TruckState
from common_lib.evaluation_utils import evaluate_solution
from common_lib.visualization_utils_plotly import create_routing_graph, visualize_routing_solution
from logisticsrl_lib.reinforcelearning.rewards import NormalizedRewards

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
    nodesObjs = data["nodes"] 
    nodes = torch.tensor([[n.lat, n.lon] for n in nodesObjs], dtype=torch.float32)    
    source_mask = np.array([getattr(node, 'isSource', False) for node in nodesObjs], dtype=bool)  
    # List, Initial truck positions (at their depots)
    truck_starts = [truck.depot_idx for truck in data["trucks"]]

    # --- Wandb Init ---
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name, 
            name=cfg.run_name, 
            config=vars(cfg)
        )

    print_verification_info(nodesObjs, data, truck_starts)   
    
    
    fleetStatus = FleetStatus(
        truck_starts=truck_starts,
        source_mask=source_mask,
        time_matrix=data["time_matrix"],
        nodes=nodes
    )
        
    rewards = NormalizedRewards(cfg,time_matrix=data["time_matrix"])    
    
    env = TSPEnv(
        cfg=cfg,
        fleetStatus=fleetStatus,
        normalized_rewards=rewards
    )


    
    agent = REINFORCEAgent(
        cfg=cfg,
        fleetStatus=fleetStatus   
    )



    # Training Loop, tqdm for a nice progress bar    
    pbar = tqdm(range(cfg.episodes))
    print(f"--> STARTING RUN: {cfg.run_name}")
    print("episode is=", cfg.episodes)
    for episode in pbar:
        obs, _ = env.reset()
        done, terminated = False, False
        episode_reward = 0.0


        while not (done or terminated):
            action = agent.act(obs)
            if cfg.debug: print(f"DEBUG: Selected action: {action}")
            obs, reward, done, terminated, _ = env.step(action)            
            agent.store_reward(reward)
            episode_reward += reward

        # Check constraints 
        total_destinations_visited, total_time, pct_intersections = evaluate_solution(env.fleetStatus.all_tours(), data, truck_starts, cfg)                
        


        loss, entropy, grad_norm, mean_normalized_return = agent.update()

        report_every_50_episodes(
            episode,
            episode_reward,
            loss,
            total_time,
            total_destinations_visited,
            pct_intersections,
            env,
            data,
            truck_starts,
            pbar)
             
                        

        # Logging to W&B
        if cfg.wandb:
            wandb.log({
                "Total reward": episode_reward,
                "Last Loss": loss,
                "Mean Entropy": entropy,
                "Episode": episode,
                "Total time": total_time,
                "Total destinations visited": total_destinations_visited,
                "Percentage of intersections": pct_intersections,
                "Mean gradient norm": grad_norm,
                "Mean normalized return": mean_normalized_return
            })
            
        pbar.set_description(f"Rw: {episode_reward:.2f}")

    # Save
    #path = f"checkpoints/{cfg.run_name}.pt"
    #torch.save(agent.policy.state_dict(), path)
    #print(f"--> SAVED: {path}")
    
    if cfg.wandb:
        wandb.finish()

def report_every_50_episodes(
    episode,
    episode_reward,
    loss,
    total_time,
    total_destinations_visited,
    pct_intersections,
    env,
    data,
    truck_starts,
    pbar
):
    if episode % 50 == 0:
        print(
            f"Episode {episode:4d} | "
            f"Total reward: {episode_reward:.3f} | "
            f"last Loss: {loss:.4f} | "
        )
        print("Total time: ", total_time)
        print("Total destinations visited: ", total_destinations_visited)
        print("Percentage of intersections: ", pct_intersections)
        pbar.write("\n--- Sample Route Plan ---")
        # total time and tour for each truck
        for i, truck_state in env.fleetStatus.trucklist.items():
            print(f"Truck {i}: total time = {truck_state.total_time:.2f}, tour = {truck_state.tour}")
        pbar.write("-------------------------\n")

        for customer in data["customers"]:
            if customer.idx in [node for tour in env.fleetStatus.all_tours() for node in tour]:
                customer.delivered = True
            else:
                customer.delivered = False 

        # Visualization
        G = create_routing_graph(data["depots"], data["customers"], env.fleetStatus.all_tours(), truck_starts)
        visualize_routing_solution(G, step=episode, title_suffix="Final step", save_path=f"checkpoints/visualization_episode{episode}.html")


def print_verification_info(nodesObjs, data, truck_starts):
    print("\n" + "="*40)
    print("🚚  DATA INSTANCE SUMMARY  🚚")
    print("="*40)
    print(f"• Number of nodes:         {len(nodesObjs)}")
    print(f"• Number of depots:        {len(data['depots'])}")
    print(f"• Number of customers:     {len(data['customers'])}")
    print(f"• Number of trucks:        {len(data['trucks'])}")
    print(f"• Truck start positions:   {truck_starts}")
    print("="*40 + "\n")


if __name__ == "__main__":
    train()