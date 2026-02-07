import os
import sys

import torch
from tqdm import tqdm

import wandb
from configs.config import parse_args
from core.envs.tsp_env import MDVRP_one_agent_per_truck_env, MDVRPEnv
from core.models.agent import (MDVRP_one_agent_per_truck_REINFORCE_agent,
                               MDVRPREINFORCEAgent)
from core.utils.data_loader import MDVRPDataLoader
from core.utils.evaluation_utils import evaluate_solution
from core.utils.visualization_utils_plotly import (create_routing_graph,
                                                   visualize_routing_solution)


def train_one_episode_one_step_all_fleet():
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
    customers = data["customers"]
    depots = data["depots"]
    truck_starts = [truck.depot_idx for truck in data["trucks"]]

    if cfg.wandb:
        wandb.init(project="mdvrp-rl", name=cfg.run_name, config=vars(cfg))

    env = MDVRPEnv(cfg, data)
    agent = MDVRPREINFORCEAgent(cfg, data)
    batch_rewards = []
    episode_reward = 0.0

    print(f"--> STARTING RUN: {cfg.run_name}")
    pbar = tqdm(range(cfg.episodes))
    for episode in pbar:
        state, _ = env.reset()
        
        # One-shot action
        # print('state ', state)
        action = agent.act(state)
        state, reward, _, _, info = env.step(action)
        total_visited = sum(len(route) for route in action.values())
        
        agent.store_reward(reward)
        batch_rewards.append(reward) # Track rewards for this batch
        episode_reward += reward
    
        if (episode + 1) % 10 == 0:
            
            # for truck_id, route in action.items():
            #     if route: # Only print trucks that actually moved
            #         print(f"Truck {truck_id}: {route}")
            
            
            loss = agent.update()
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            pbar.write(
                    f"Episode {episode+1:>4} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Visited: {total_visited:.2f} | "
                    f"Loss: {loss:.4f} | "
                    f"Time: {info['total_time']:.2f}h"
                )
            
            if cfg.wandb:
                wandb.log({
                    "episode ":episode,
                    "batch_avg_reward ": avg_reward, 
                    "visited ": total_visited, 
                    "loss ": loss, 
                    "total_time ": info["total_time"]
                })
            batch_rewards = [] # Reset for next batch
            
            G = create_routing_graph(depots, customers, env.tours, truck_starts)
            visualize_routing_solution(G, step=episode, title_suffix="Final step", save_path=f"checkpoints/visualization_episode{episode}.html")
        
            

    torch.save(agent.policy.state_dict(), f"checkpoints/mdvrp_{cfg.run_name}.pt")
    if cfg.wandb: wandb.finish()





def train_truck_by_truck():

    
    cfg = parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    
    loader = MDVRPDataLoader(data_dir=cfg.data_dir)
    data = loader.load_data()
    customers = data["customers"]
    depots = data["depots"]
    truck_starts = [truck.depot_idx for truck in data["trucks"]]

    if cfg.wandb:
        wandb.init(project="mdvrp-rl", name=cfg.run_name, config=vars(cfg))

    env = MDVRP_one_agent_per_truck_env(cfg, data, truck_starts)
    agent = MDVRP_one_agent_per_truck_REINFORCE_agent(cfg, data)
    
    batch_rewards = []
    episode_reward = 0
    print(f"--> STARTING RUN: {cfg.run_name}")
    
    pbar = tqdm(range(cfg.episodes))
    for episode in pbar:
        env.use_2opt = (episode == cfg.episodes - 1)
        state, _ = env.reset()
        terminated = False
        
        episode_reward = 0
        episode_steps = 0
        
        # --- Step-by-Step Loop ---
        while not terminated:
            
    
            # WHO & WHERE: The state tells the agent: "Truck #5 is at Node A and ready." - truck by truck-
            # DECISION: Agent picks a CUSTOMER for Truck #5. 
            # Before picking, the agent calculates a constraint_mask. 
            # It "sees" which customers are too far to visit and still get home by the 24h mark. 
            # It effectively ignores those customers, making it impossible to pick an invalid move
            action = agent.act(state) 
            
            
            # FEEDBACK: Store the reward for that specific move.
            if action is not None:
                # MOVEMENT: Environment "drives" Truck #5 to the customer.
                # UPDATE: Environment marks customer as visited and updates Truck #5's clock.
                # NEXT UP: Environment finds the next truck that will be free and puts it in the 'state'.
                state, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    # Objective 1 & 2: Validate results
                    total_visited = info.get('total_visited', episode_steps)
                    total_time = info.get('optimized_total_time', 0.0)  
                agent.store_reward(reward) # Only store if we acted
                episode_reward += reward
                        
            episode_steps += 1            
             
        batch_rewards.append(episode_reward)

        # --- Batch Update ---
        if (episode + 1) % 10 == 0:
            loss = agent.update()
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            
            # Extract truck-specific data from the LAST episode of the batch
            truck_results = info.get("truck_results", {})
            active_trucks = [s for s in truck_results.values() if s["route"]]
            num_active = len(active_trucks)
            
            # Calculate average time only for trucks that actually worked
            avg_work_time = sum(s["time"] for s in active_trucks) / num_active if num_active > 0 else 0
            max_work_time = max([s["time"] for s in active_trucks], default=0)

            
            # Update Progress Bar with current metrics
            pbar.write(
                f"Episode {episode+1:>4} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Visited: {total_visited} | "
                f"Loss: {loss:.4f} | "
                f"Total Time: {total_time:.2f}h |"
                f"Trucks: {num_active:>2} | "
                f"AvgT: {avg_work_time:.1f}h | "
                f"MaxT: {max_work_time:.1f}h"
            )
            
            if cfg.wandb:
                wandb_data = {
                    "avg_reward": avg_reward,
                    "total_visited": total_visited,
                    "loss": loss,
                    "fleet/active_trucks": num_active,
                    "fleet/avg_truck_time": avg_work_time,
                    "fleet/max_truck_time": max_work_time,
                    "fleet/total_fleet_time": total_time,
                }
                
                # Log distribution of times (creates a histogram in wandb)
                if active_trucks:
                    times = [s["time"] for s in active_trucks]
                    wandb_data["distributions/truck_times"] = wandb.Histogram(times)
                
                wandb.log(wandb_data)
            
            
            batch_rewards = [] # Reset batch tracker
            
            G = create_routing_graph(depots, customers, env.tours, truck_starts)
            visualize_routing_solution(G, step=episode, title_suffix="Final step", save_path=f"checkpoints/visualization_episode{episode}.html")
        
            

            # Optional: Save Checkpoint
            if (episode + 1) % 50 == 0:
                pbar.write("\n--- Sample Route Plan ---")
                for tid, s in list(truck_results.items()): # Show all trucks trucks
                    if s["route"]:
                        pbar.write(f"  T{tid:02}: {len(s['route']):>2} stops | {s['time']:5.1f}h | {s['route']}")
                pbar.write("-------------------------\n")

    if cfg.wandb:
        wandb.finish()
        
if __name__ == "__main__":
    train_truck_by_truck() 
    # train_one_episode_one_step_all_fleet()