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
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_knn(time_matrix: torch.Tensor, knn_k: int, device):
    """
    Precompute KNN graph from the travel-time matrix.
    time_matrix[i, j] = travel time from node i to node j.
    Returns:
        edge_index:    [2, E] LongTensor on `device` — used by FactorizedFleetPolicy (GNN)
        knn_neighbors: list of sets of length N       — used by NormalizedRewards (zone bonus)
    Both use the same distance metric (travel time), so they reference the same clusters.
    """
    dist = time_matrix.float().to(device)               # [N, N]
    num_nodes = dist.shape[0]

    # Exclude self (diagonal = inf so it's never picked as a neighbor)
    dist = dist.clone()
    dist.fill_diagonal_(float('inf'))
    knn = dist.topk(knn_k, largest=False).indices        # [N, knn_k]

    # edge_index (bidirectional + self-loops)
    src = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, knn_k).reshape(-1)
    dst = knn.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    self_loops = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    # knn_neighbors: list of sets (CPU, for reward lookup)
    knn_cpu = knn.cpu().tolist()
    knn_neighbors = [set(row) for row in knn_cpu]

    return edge_index, knn_neighbors



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

    # Precompute KNN once — shared by policy (GNN edges) and rewards (zone bonus)
    edge_index, knn_neighbors = build_knn(data["time_matrix"], knn_k=15, device=cfg.device)

    env = TSPEnv(
        cfg=cfg,
        truck_starts=truck_starts,
        source_mask=source_mask,
        time_matrix=data["time_matrix"],
        nodes=nodes,
        knn_neighbors=knn_neighbors
    )

    agent = REINFORCEAgent(cfg, edge_index=edge_index)

    # Training Loop, tqdm for a nice progress bar    
    pbar = tqdm(range(cfg.episodes))
    print(f"--> STARTING RUN: {cfg.run_name}")
    print("episode is=", cfg.episodes)
    max_reward = -float('inf')
    for episode in pbar:
        obs, _ = env.reset()
        done, terminated = False, False
        episode_reward = 0.0
        


        while not (done or terminated):
            action = agent.act(obs)
            if cfg.debug: print(f"DEBUG: Selected action: {action}")
            obs, reward, done, terminated, info = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            if done or terminated:
                agent.store_terminal_bonus(info.get("terminal_bonus", 0.0))

        # Check constraints 
        total_destinations_visited, total_time, pct_intersections = evaluate_solution(env.fleetStatus.all_tours(), data, truck_starts, cfg)                
        


        loss, entropy, grad_norm, mean_normalized_return = agent.update()

        final_reward = total_destinations_visited - total_time/20
        if final_reward > max_reward:
            max_reward = final_reward
            print(f"New best solution found! Total reward: {final_reward:.2f}, Total time: {total_time:.2f}, Total destinations visited: {total_destinations_visited}, Percentage of intersections: {pct_intersections:.2f}%")
            report(
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
                cfg,
                agent
            )

        report(
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
            cfg,
            agent
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

def report(
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
    cfg,
    agent
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

    # Save model:
    path = f"checkpoints/{cfg.run_name}_best.pt"
    torch.save(agent.policy.state_dict(), path)
    print(f"--> SAVED BEST MODEL: {path}")

def print_verification_info(nodesObjs, data, truck_starts):
    print("\n" + "="*40)
    print("   DATA INSTANCE SUMMARY")
    print("="*40)
    print(f"• Number of nodes:         {len(nodesObjs)}")
    print(f"• Number of depots:        {len(data['depots'])}")
    print(f"• Number of customers:     {len(data['customers'])}")
    print(f"• Number of trucks:        {len(data['trucks'])}")
    print(f"• Truck start positions:   {truck_starts}")
    print("="*40 + "\n")


if __name__ == "__main__":
    train()