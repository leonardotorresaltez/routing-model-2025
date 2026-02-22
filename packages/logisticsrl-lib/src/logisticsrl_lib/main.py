import os
import random
import sys

import numpy as np
import torch
from common_lib.evaluation_utils import evaluate_solution
from common_lib.visualization_utils_plotly import (create_routing_graph,
                                                   visualize_routing_solution)
from loader_lib.data_loader import FleetStatus, MDVRPDataLoader, TruckState
from logisticsrl_lib.configs.config import parse_args
from logisticsrl_lib.reinforcelearning.agent import PPOAgent
from logisticsrl_lib.reinforcelearning.tsp_env import TSPEnv
from tqdm import tqdm

import wandb


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def train():
    cfg = parse_args()
    set_seed(cfg.seed)
    
    loss = 0.0  # Initialize loss outside loop
    best_reward = 0.0
    
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
        active_truck=0,
        time_matrix=data["time_matrix"],
        nodes=nodes
    )
        
    env = TSPEnv(
        cfg=cfg,
        fleetStatus=fleetStatus   
    )

    agent = PPOAgent(
        cfg=cfg,
        fleetStatus=fleetStatus   
    )

    # Training Loop, tqdm for a nice progress bar    
    pbar = tqdm(range(cfg.episodes))
    print(f"--> STARTING RUN: {cfg.run_name}")
    print("episode is=", cfg.episodes)
    for episode in pbar:
        obs, _ = env.reset()
        episode_reward = 0.0

        while True:
            action = agent.act(obs)
            
            if action == -1:
                # Force trucks back to depot to get accurate final time and tour completion
                for t_id in range(env.num_trucks):
                    truck = env.fleetStatus.trucklist[t_id]
                    depot_idx = env.fleetStatus.truck_starts[t_id]
                    if truck.position != depot_idx:
                        dist_to_home = env.fleetStatus.time_matrix[truck.position, depot_idx]
                        truck.total_time += dist_to_home
                        truck.position = depot_idx
                        truck.tour.append(depot_idx)
                        
                final_bonus = env._calculate_episode_reward()
                
                if len(agent.rewards) > 0:
                    agent.rewards[-1] += final_bonus
                episode_reward += final_bonus
                break
            
            # if action == -1:
            #     # Agent ran out of time/trucks. We break without calling step().
            #     # No episode-end bonus needed anymore, the dense step rewards handle it.
            #     final_bonus = env._calculate_episode_reward()
            #     # ADD bonus to the last action's reward instead of appending a new one
            #     if len(agent.rewards) > 0:
            #         agent.rewards[-1] += final_bonus
            #     episode_reward += final_bonus
            #     num_delivered = env.visited_targets[env.target_mask].sum()
            #     total_time = sum(env.fleetStatus.trucklist[t_id].total_time for t_id in range(env.num_trucks))
            #     # print(f" END: delivered={num_delivered}, time={total_time:.2f}h  ")
            #     break
            
            obs, reward, done, terminated, _ = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            
            if done or terminated:
                break
            
        
        frac = 1.0 - (episode / cfg.episodes)# Calculate the fraction of training remaining (goes from 1.0 down to 0.0)        
        new_lr = max(1e-6, cfg.lr * frac) # Calculate new learning rate, keeping a tiny minimum floor (e.g., 1e-6) so it never completely stops learning
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Check constraints 
        total_destinations_visited, total_time, pct_intersections = evaluate_solution(env.fleetStatus.all_tours(), data, truck_starts, cfg)                

        
         
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Save the best model
            torch.save(agent.policy.state_dict(), "checkpoints/best_ppo_model.pt")
        
        
        
        should_step = True
        loss_val, entropy, val_loss, mean_return = agent.update()

        if should_step and (episode % 50 == 0):
            tqdm.write(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Time: {total_time:5.1f}h | "
                f"Visited {total_destinations_visited:4d} | "
                f"Critic Loss (Val): {val_loss:6.2f} | "
                f"Entropy: {entropy:5.3f}"
            )
        
        pbar.set_description(f"Rw: {episode_reward:.2f}")
        
        if should_step:
            loss = loss_val
            
            # Print to console and generate HTML
            report_every_N_episodes(
                episode,
                episode_reward,
                loss,
                total_time,
                total_destinations_visited,
                pct_intersections,
                env,
                data,
                truck_starts,
                pbar,
                cfg
            ) 

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
                    "Mean gradient norm": val_loss,
                    "Mean normalized return": mean_return
                })

    # Save
    #path = f"checkpoints/{cfg.run_name}.pt"
    #torch.save(agent.policy.state_dict(), path)
    #print(f"--> SAVED: {path}")
    
    if cfg.wandb:
        wandb.finish()

def report_every_N_episodes(
    episode,
    episode_reward,
    loss,
    total_time,
    total_destinations_visited,
    pct_intersections,
    env,
    data,
    truck_starts,
    pbar,
    cfg
):
    # Handle loss being a tuple (extract first element if needed)
    if isinstance(loss, tuple):
        loss = loss[0]
        

    if episode % (cfg.update_every * 500) == 0:
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