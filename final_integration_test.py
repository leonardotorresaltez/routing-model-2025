"""
Final integration test for the complete fleet routing system.

Tests:
1. Data generation
2. Environment creation and reset
3. Baseline policy execution
4. PPO model loading and evaluation
5. End-to-end workflow
"""

import os
import numpy as np
from env.routing_env_simple import SimpleFleetRoutingEnv
from env.sb3_wrapper import FleetRoutingSB3Wrapper
from env.data_generation import generate_problem_instance, calculate_optimal_bounds
from baselines import greedy_nearest_customer_policy
from stable_baselines3 import PPO

def test_complete_workflow():
    """Run a complete workflow test."""
    print("\n" + "=" * 70)
    print("FINAL INTEGRATION TEST - COMPLETE FLEET ROUTING SYSTEM")
    print("=" * 70 + "\n")
    
    print("[TEST 1] Data Generation")
    print("-" * 70)
    customers, trucks, depots = generate_problem_instance(
        num_customers=15,
        num_trucks=4,
        num_depots=2,
        seed=123
    )
    print(f"Generated problem: {len(customers)} customers, {len(trucks)} trucks, {len(depots)} depots")
    
    try:
        lower_bound, upper_bound = calculate_optimal_bounds(customers, trucks, depots)
        print(f"Optimal bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print("[PASS] Data generation\n")
    except Exception as e:
        print(f"[FAIL] Data generation: {e}\n")
        return False
    
    print("[TEST 2] Environment Creation and Reset")
    print("-" * 70)
    try:
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=200)
        obs, info = env.reset()
        print(f"Environment created with observation shape: {obs['truck_positions'].shape}")
        print(f"Observation keys: {list(obs.keys())}")
        print("[PASS] Environment creation and reset\n")
    except Exception as e:
        print(f"[FAIL] Environment creation: {e}\n")
        return False
    
    print("[TEST 3] SB3 Wrapper")
    print("-" * 70)
    try:
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=200)
        wrapped_env = FleetRoutingSB3Wrapper(env)
        obs, info = wrapped_env.reset()
        print(f"Wrapped observation shape: {obs.shape}")
        print(f"Observation space: {wrapped_env.observation_space}")
        print("[PASS] SB3 wrapper\n")
    except Exception as e:
        print(f"[FAIL] SB3 wrapper: {e}\n")
        return False
    
    print("[TEST 4] Baseline Policy Execution")
    print("-" * 70)
    try:
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=200)
        metrics = greedy_nearest_customer_policy(env)
        print(f"Baseline results:")
        print(f"  Total distance: {metrics['total_distance']:.2f}")
        print(f"  Customers delivered: {metrics['customers_delivered']}/{len(customers)}")
        print(f"  Episode length: {metrics['episode_length']}")
        print(f"  Avg utilization: {metrics['avg_utilization']:.2%}")
        print("[PASS] Baseline policy execution\n")
    except Exception as e:
        print(f"[FAIL] Baseline execution: {e}\n")
        return False
    
    print("[TEST 5] PPO Model Loading and Inference")
    print("-" * 70)
    try:
        model_path = "models/ppo_routing_final"
        if os.path.exists(f"{model_path}.zip"):
            model = PPO.load(model_path)
            env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=200)
            wrapped_env = FleetRoutingSB3Wrapper(env)
            
            obs, _ = wrapped_env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            print(f"Model inference results:")
            print(f"  Episode length: {steps}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Total distance: {info['total_distance']:.2f}")
            print(f"  Customers delivered: {info['customers_delivered']}/{len(customers)}")
            print("[PASS] PPO model loading and inference\n")
        else:
            print("Skipping PPO test (model not found)")
            print("[SKIP] PPO model test\n")
    except Exception as e:
        print(f"[FAIL] PPO inference: {e}\n")
        return False
    
    print("[TEST 6] Feasibility Masking")
    print("-" * 70)
    try:
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=100)
        obs, _ = env.reset()
        
        for step in range(10):
            feasibility_mask = obs["feasibility_mask"]
            unvisited_mask = obs["unvisited_mask"]
            
            all_visited = set()
            for ts in env.truck_states:
                all_visited.update(ts.visited_customers)
            
            for truck_idx in range(len(env.trucks)):
                for customer_idx in range(len(customers)):
                    is_feasible = feasibility_mask[truck_idx, customer_idx] == 1
                    is_visited = customer_idx in all_visited
                    
                    if is_feasible and is_visited:
                        print(f"[ERROR] Truck {truck_idx} marked feasible for visited customer {customer_idx}")
                        return False
            
            action = np.zeros(len(trucks), dtype=int)
            for truck_idx in range(len(trucks)):
                feasible_actions = np.where(feasibility_mask[truck_idx] == 1)[0]
                if len(feasible_actions) > 0:
                    action[truck_idx] = np.random.choice(feasible_actions)
                else:
                    action[truck_idx] = len(customers)
            
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        print(f"Feasibility constraints validated across {step + 1} steps")
        print("[PASS] Feasibility masking\n")
    except Exception as e:
        print(f"[FAIL] Feasibility masking: {e}\n")
        return False
    
    print("=" * 70)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 70)
    print("\nSystem is fully operational and ready for training.")
    print("Run 'python train_ppo.py' to start training a policy.")
    print("\nFor detailed information, see TRAINING_GUIDE.md")
    return True


if __name__ == "__main__":
    success = test_complete_workflow()
    exit(0 if success else 1)
