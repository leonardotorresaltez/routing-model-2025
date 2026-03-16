import torch
import numpy as np
import pickle
import time
import os
import argparse
from tqdm import tqdm
import pandas as pd

from logisticsrl_lib.reinforcelearning.tsp_env import TSPEnv
from logisticsrl_lib.compute_benchmarks_google_ortools import solve_with_ortools
from common_lib.evaluation_utils import evaluate_solution
from loader_lib.data_loader import MDVRPDataLoader

def subsample_data(original_data, num_target_nodes):
    """
    Subsamples the original data to keep only the first N nodes.
    Maintains depots (which are at the beginning of the node list).
    """
    if num_target_nodes >= original_data["num_nodes"]:
        return original_data, [t.depot_idx for t in original_data["trucks"]]

    new_data = original_data.copy()
    
    # 1. Slice nodes and time matrix
    new_data["nodes"] = original_data["nodes"][:num_target_nodes]
    new_data["time_matrix"] = original_data["time_matrix"][:num_target_nodes, :num_target_nodes]
    new_data["num_nodes"] = num_target_nodes
    
    # 2. Filter depots and customers
    new_data["depots"] = [d for d in original_data["depots"] if d.idx < num_target_nodes]
    new_data["customers"] = [c for c in original_data["customers"] if c.idx < num_target_nodes]
    
    # 3. Filter trucks and identify truck starts
    # Only keep trucks that start at a depot that is still in the environment
    new_data["trucks"] = [t for t in original_data["trucks"] if t.depot_idx < num_target_nodes]
    truck_starts = [t.depot_idx for t in new_data["trucks"]]
    
    # 4. Update node features (re-normalization for the subsampled matrix)
    time_tensor = new_data["time_matrix"]
    new_data["node_features"] = time_tensor / (time_tensor.max() + 1e-9)
    
    return new_data, truck_starts

def run_rl_inference(agent, env, data, truck_starts, cfg):
    start_time = time.time()
    obs, _ = env.reset()
    done, terminated = False, False
    
    while not (done or terminated):
        # The agent's act method uses sampling. 
        # For evaluation, we could potentially use argmax on probabilities if we wanted greedy,
        # but here we follow the standard act() which includes the policy forward pass.
        with torch.no_grad():
            action = agent.act(obs)
        obs, reward, done, terminated, info = env.step(action)
    
    compute_time = time.time() - start_time
    
    tours = env.fleetStatus.all_tours()
    total_destinations_visited, total_time, _ = evaluate_solution(tours, data, truck_starts, cfg)
    
    return compute_time, total_destinations_visited, total_time

def run_ortools_benchmark(time_matrix, truck_starts, max_time, data, cfg, time_limit):
    start_time = time.time()
    # OR-Tools expects integer costs
    TIME_SCALE = 100
    scaled_time_matrix = (time_matrix * TIME_SCALE).round().long()
    scaled_max_time = int(max_time * TIME_SCALE)
    
    tours = solve_with_ortools(scaled_time_matrix, truck_starts, scaled_max_time, time_limit=time_limit)
    compute_time = time.time() - start_time
    
    total_destinations_visited, total_time, _ = evaluate_solution(tours, data, truck_starts, cfg)
    
    return compute_time, total_destinations_visited, total_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="pointer_network_a2c", help="Run name used during training to find the best agent pickle")
    parser.add_argument("--node_counts", type=int, nargs="+", default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], help="List of node counts to test (subsampling from original)")
    parser.add_argument("--data_dir", type=str, default="data_version_2", help="Data directory to load base environment from")
    args = parser.parse_args()
    
    agent_path = f"checkpoints/{args.run_name}_best_agent.pkl"
    
    if not os.path.exists(agent_path):
        print(f"Error: Agent file not found at {agent_path}")
        return

    print(f"Loading agent from {agent_path}...")
    with open(agent_path, "rb") as f:
        agent = pickle.load(f)
    
    # Ensure agent is in eval mode
    if hasattr(agent, 'policy'):
        agent.policy.eval()
        agent.cfg.debug = False 
    
    cfg = agent.cfg
    
    print(f"Loading base environment data from {args.data_dir}...")
    loader = MDVRPDataLoader(data_dir=args.data_dir)
    base_data = loader.load_data()
    total_base_nodes = base_data["num_nodes"]
    print(f"Base environment has {total_base_nodes} nodes.")

    results = []
    
    for n in args.node_counts:
        if n > total_base_nodes:
            print(f"Skipping {n} nodes as it exceeds base environment size ({total_base_nodes})")
            continue
            
        print(f"\n--- Subsampling Environment to {n} nodes ---")
        sub_data, truck_starts = subsample_data(base_data, n)
        
        # Extract features for TSPEnv
        # Note: TSPEnv uses nodes, truck_starts, source_mask, time_matrix
        nodes_coords = torch.tensor([[n.lat, n.lon] for n in sub_data["nodes"]], dtype=torch.float32)
        source_mask = np.array([node.isSource for node in sub_data["nodes"]], dtype=bool)
        
        env = TSPEnv(
            cfg=cfg,
            truck_starts=truck_starts,
            source_mask=source_mask,
            time_matrix=sub_data["time_matrix"],
            nodes=nodes_coords
        )
        
        print(f"Running RL Inference on {n} nodes...")
        # Run RL inference multiple times if needed for stability? 
        # For now, 1 run.
        rl_comp_time, rl_visited, rl_total_time = run_rl_inference(agent, env, sub_data, truck_starts, cfg)
        
        print(f"Running OR-Tools Benchmark (10s) on {n} nodes...")
        ot10_comp_time, ot10_visited, ot10_total_time = run_ortools_benchmark(
            sub_data["time_matrix"], 
            truck_starts, 
            cfg.max_daily_delivery_time_each_truck, 
            sub_data, 
            cfg,
            time_limit=10
        )

        print(f"Running OR-Tools Benchmark (60s) on {n} nodes...")
        ot60_comp_time, ot60_visited, ot60_total_time = run_ortools_benchmark(
            sub_data["time_matrix"], 
            truck_starts, 
            cfg.max_daily_delivery_time_each_truck, 
            sub_data, 
            cfg,
            time_limit=60
        )
        
        results.append({
            "Nodes": n,
            "RL_Time": f"{rl_comp_time:.3f}s",
            "RL_Visited": rl_visited,
            "RL_TotalTravel": f"{rl_total_time:.2f}",
            "OT10_Visited": ot10_visited,
            "OT10_TotalTravel": f"{ot10_total_time:.2f}",
            "OT60_Visited": ot60_visited,
            "OT60_TotalTravel": f"{ot60_total_time:.2f}"
        })
    
    # Print Comparison Table
    df = pd.DataFrame(results)
    print("\nComparison Results (Subsampled Environments):")
    print(df.to_string(index=False))
    
    # Save results to CSV
    output_path = "subsampled_inference_benchmark.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
