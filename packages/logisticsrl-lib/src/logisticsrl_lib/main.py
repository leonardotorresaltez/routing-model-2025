import torch
import random
import numpy as np
import os
import wandb
from tqdm import tqdm
import sys

from logisticsrl_lib.configs.config import parse_args
from logisticsrl_lib.reinforcelearning.tsp_env import TSPEnv
from logisticsrl_lib.reinforcelearning.agent import PPOAgent
from loader_lib.data_loader import FleetStatus, MDVRPDataLoader, TruckState
from common_lib.evaluation_utils import evaluate_solution
from common_lib.curriculum_learning_utils import get_curriculum_iterator
from common_lib.visualization_utils_plotly import create_routing_graph, visualize_routing_solution

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
    truck_starts = [truck.depot_idx for truck in data["trucks"]]
    n_total_trucks = len(data["trucks"])
    curriculum = get_curriculum_iterator(start_nodes = 5, n_total_nodes = data["num_nodes"], ratio_trucks_nodes = n_total_trucks / data["num_nodes"])
    current_n_nodes, current_n_trucks = next(curriculum)
    n_successes = 0
    max_reward = 0.0

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

    # ---> NEW: Instantiate PPOAgent
    agent = PPOAgent(
        cfg=cfg,
        fleetStatus=fleetStatus   
    )

    last_tours = ""
    no_change_count = 0
    pbar = tqdm(range(cfg.episodes))
    print(f"--> STARTING RUN: {cfg.run_name}")
    print("episode is=", cfg.episodes)
    
    for episode in pbar:
        obs, _ = env.reset(n_nodes=current_n_nodes, n_trucks=current_n_trucks)
        done, terminated = False, False
        episode_reward = 0.0

        while not (done or terminated):
            action = agent.act(obs)
            if getattr(cfg, 'debug', False): print(f"DEBUG: Selected action: {action}")
            obs, reward, done, terminated, _ = env.step(action)  
            is_terminal = done or terminated          
            agent.store_reward(reward, is_terminal)
            episode_reward += reward

        total_destinations_visited, total_time, pct_intersections = evaluate_solution(env.fleetStatus.all_tours(), data, truck_starts, cfg)                
        if total_destinations_visited > current_n_nodes + len(data['depots']):
            print("Current tours:", env.fleetStatus.all_tours())
            raise ValueError(f"Error: Visited more destinations ({total_destinations_visited}) than the current curriculum allows (max {current_n_nodes + len(data['depots'])}). Check the curriculum reset logic.")
        
        if episode_reward > max_reward and current_n_nodes == data["num_nodes"]:
            max_reward = episode_reward
            print(f"New max reward: {max_reward:.2f} at episode {episode}")
            report(episode, episode_reward, loss if 'loss' in locals() else 0.0, total_time, total_destinations_visited,
                   pct_intersections, env, data, truck_starts, pbar, current_n_nodes, current_n_trucks)

        if pct_intersections < 0.001 and env.noop_count < current_n_trucks: # If we have a perfect solution, advance the curriculum
            n_successes += 1
            if n_successes >= cfg.curriculum_learning_successes_required: # Require n successful episodes before advancing curriculum (to avoid flukes)
                n_successes = 0
                report(episode, episode_reward, loss if 'loss' in locals() else 0.0, total_time, total_destinations_visited,
                       pct_intersections, env, data, truck_starts, pbar, current_n_nodes, current_n_trucks)

                print("-"*50)
                print(f"Curriculum advanced! Now training on {current_n_nodes} nodes and {current_n_trucks} trucks.")
                print("-"*50)
                current_n_nodes, current_n_trucks = next(curriculum)
        else:
            # n_successes = 0  # We want consecutive successes, so reset the count if we fail
            pass

        if str(env.fleetStatus.all_tours()) == last_tours:
            no_change_count += 1
            limit = 51
            if no_change_count >= limit:
                print(f"No improvement in tours for {limit} episodes. Terminating training.")
                break
        else:
            no_change_count = 0
        last_tours = str(env.fleetStatus.all_tours())

        # Update happens every X episodes, handling the full batched PPO loop internally!
        if episode % cfg.episodes_per_update_batch == 0 and episode > 0:
            (loss,
            entropy,
            grad_norm,
            explained_var,
            approx_kl,
            clip_frac,
            adv_std,
            noop_prob
            ) = agent.update()

        

        if cfg.wandb and episode % cfg.log_interval == 0:
        # if cfg.wandb and episode % 1 == 0:
            wandb.log({
                "Total reward": episode_reward,
                "Last Loss": loss if 'loss' in locals() else 0.0,
                "Mean Entropy": entropy if 'entropy' in locals() else 0.0,
                "Episode": episode,
                "Total time": total_time,
                "Total destinations visited": total_destinations_visited,
                "Percentage of intersections": pct_intersections,
                "Mean gradient norm": grad_norm if 'grad_norm' in locals() else 0.0,
                "NO-OP count": env.noop_count,
                "Explained variance": explained_var if 'explained_var' in locals() else 0.0,
                "Approx KL": approx_kl if 'approx_kl' in locals() else 0.0,
                "Clip fraction": clip_frac if 'clip_frac' in locals() else 0.0,
                "Advantage std": adv_std if 'adv_std' in locals() else 0.0,
                "NOOP probability": noop_prob if 'noop_prob' in locals() else 0.0,
            })
            
        pbar.set_description(f"Rw: {episode_reward:.2f}")
    
    if cfg.wandb:
        wandb.finish()

def report(
    episode, episode_reward, loss, total_time, total_destinations_visited,
    pct_intersections, env, data, truck_starts, pbar, current_n_nodes, current_n_trucks
):
        print(
            f"Episode {episode:4d} | "
            f"Total reward: {episode_reward:.3f} | "
            f"last Loss: {loss:.4f} | "
        )
        print("Total time: ", total_time)
        print("Total destinations visited: ", total_destinations_visited)
        print("Percentage of intersections: ", pct_intersections)
        pbar.write("\n--- Sample Route Plan ---")

        for i, truck_state in env.fleetStatus.trucklist.items():
            print(f"Truck {i}: total time = {truck_state.total_time:.2f}, tour = {truck_state.tour}")
        pbar.write("-------------------------\n")

        for customer in data["customers"]:
            if customer.idx in [node for tour in env.fleetStatus.all_tours() for node in tour]:
                customer.delivered = True
            else:
                customer.delivered = False 

        G = create_routing_graph(data["depots"], data["customers"], env.fleetStatus.all_tours(), truck_starts)
        visualize_routing_solution(G, step=episode, title_suffix="Final step", save_path=f"checkpoints/visualization_nnodes{current_n_nodes}_ntrucks{current_n_trucks}_reward{int(episode_reward*10)}.html")

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