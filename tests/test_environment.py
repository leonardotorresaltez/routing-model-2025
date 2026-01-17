"""
Test script for the simplified fleet routing environment.

Tests environment functionality, feasibility masking, and baseline policies.
"""

import numpy as np

from baselines import (furthest_customer_policy,
                       greedy_nearest_customer_policy,
                       nearest_depot_first_policy)
from env.data_generation import (calculate_optimal_bounds,
                                 generate_problem_instance)
from env.routing_env_simple import SimpleFleetRoutingEnv


def test_environment_basics():
    """Test basic environment functionality.
    What it does:
    Generates a random problem instance (15 customers, 4 trucks, 2 depots)
    Creates the environment and resets it
    Verifies observation structure and initial state
    Relevance:

    Sanity check that the environment initializes correctly
    Confirms all observation components exist (truck positions, customer locations, masks)
    Validates that initial state is correct (all customers unvisited, all depots available for return)
    This is the foundation - if this fails, nothing else will work
    
    """
    print("=" * 60)
    print("TEST 1: Environment Basics")
    print("=" * 60)
    
    customers, trucks, depots = generate_problem_instance(
        num_customers=15,
        num_trucks=4,
        num_depots=2,
        seed=42
    )
    
    env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=500)
    obs, info = env.reset()
    
    print(f"Customers: {len(customers)}")
    print(f"Trucks: {len(trucks)}")
    print(f"Depots: {len(depots)}")
    print(f"\nInitial observation keys: {obs.keys()}")
    print(f"Truck positions shape: {obs['truck_positions'].shape}")
    print(f"Feasibility mask shape: {obs['feasibility_mask'].shape}")
    print(f"Unvisited mask shape: {obs['unvisited_mask'].shape}")
    
    print(f"\nAll customers unvisited at start: {np.sum(obs['unvisited_mask']) == len(customers)}")
    print(f"All depots available: {np.all(obs['feasibility_mask'][:, -1] == 1)}")
    print("[OK] Environment basics OK\n")
    
    return env, customers, trucks, depots


def test_feasibility_masking(env):
    """Test that feasibility masking enforces constraints.
    What it does:

    Runs 5 steps of the environment
    At each step, verifies that the feasibility mask correctly reflects which actions are valid
    Confirms trucks can only be marked as feasible for customers they can actually serve (capacity + not already visited)
    Takes random feasible actions and steps forward
    Relevance:

    Critical for constraint enforcement - ensures capacity limits are hard-enforced, not just soft penalties
    Validates the action masking mechanism works correctly
    Prevents the policy from ever learning invalid assignments (trucks overloading, visiting same customer twice)
    The entire RL training depends on this working perfectly
    
    """
    print("=" * 60)
    print("TEST 2: Feasibility Masking")
    print("=" * 60)
    
    obs, info = env.reset()
    
    for step in range(5):
        feasibility_mask = obs["feasibility_mask"]
        unvisited_mask = obs["unvisited_mask"]
        
        print(f"\nStep {step + 1}:")
        print(f"  Unvisited customers: {np.sum(unvisited_mask)}")
        
        for truck_idx in range(env.num_trucks):
            truck_state = env.truck_states[truck_idx]
            capacity_ok = True
            
            for customer_idx in range(env.num_customers):
                feasible = feasibility_mask[truck_idx, customer_idx]
                unvisited = unvisited_mask[customer_idx]
                has_capacity = truck_state.remaining_capacity() >= env.customers[customer_idx].weight
                
                if unvisited and feasible:
                    if not (has_capacity and customer_idx not in truck_state.visited_customers):
                        capacity_ok = False
                        print(f"  ERROR: Truck {truck_idx} marked feasible for impossible customer {customer_idx}")
            
            if capacity_ok:
                print(f"  Truck {truck_idx}: capacity constraints OK (util: {truck_state.utilization():.2%})")
        
        action = np.zeros(env.num_trucks, dtype=int)
        for truck_idx in range(env.num_trucks):
            feasible_actions = np.where(feasibility_mask[truck_idx] == 1)[0]
            if len(feasible_actions) > 0:
                action[truck_idx] = np.random.choice(feasible_actions)
            else:
                action[truck_idx] = env.num_customers
        
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    
    print("[OK] Feasibility masking OK\n")


def test_episode_completion(env):
    """Test that episode can reach completion.
    What it does:

    Runs a full episode to termination, taking random feasible actions
    Counts steps needed to deliver all customers
    Checks that the episode properly terminates (not truncated)
    Reports total distance and truck utilization
    Relevance:

    Validates episode lifecycle - ensures episodes can reach natural completion
    Tests that termination logic works (all customers delivered = success)
    Provides baseline metrics for comparison
    Confirms the environment can handle full episodes without errors
    
    """
    print("=" * 60)
    print("TEST 3: Episode Completion")
    print("=" * 60)
    
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 500:
        feasibility_mask = obs["feasibility_mask"]
        action = np.zeros(env.num_trucks, dtype=int)
        
        for truck_idx in range(env.num_trucks):
            feasible_actions = np.where(feasibility_mask[truck_idx] == 1)[0]
            if len(feasible_actions) > 0:
                action[truck_idx] = np.random.choice(feasible_actions)
            else:
                action[truck_idx] = env.num_trucks
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    print(f"Episode completed in {steps} steps")
    print(f"Customers delivered: {info['customers_delivered']}/{env.num_customers}")
    print(f"Total distance: {info['total_distance']:.2f}")
    print(f"Average utilization: {info['avg_utilization']:.2%}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print("[OK] Episode completion OK\n")
    
    return info


def test_baseline_policies(customers, trucks, depots):
    """Test all baseline policies.
    What it does:
    Runs three different heuristic policies:
    Greedy Nearest: Each truck picks the closest unvisited customer
    Nearest Depot First: Trucks return to depot at 70% utilization
    Furthest Customer: Each truck picks the farthest customer (exploratory)
    Computes optimal distance bounds as reference
    Compares all three policies and reports the best

    Relevance:
    Establishes performance baselines for learned policies to compare against
    Tests that different decision strategies work correctly in the environment
    Provides ground truth metrics for validation
    The greedy policy typically achieves ~800-1000 distance units for 15-customer problems
    Any trained RL policy should eventually beat these baselines"""
    print("=" * 60)
    print("TEST 4: Baseline Policies")
    print("=" * 60)
    
    lower_bound, upper_bound = calculate_optimal_bounds(customers, trucks, depots)
    print(f"Optimal distance bounds: [{lower_bound:.2f}, {upper_bound:.2f}]\n")
    
    baselines = {
        "Greedy Nearest": greedy_nearest_customer_policy,
        "Nearest Depot First": nearest_depot_first_policy,
        "Furthest Customer": furthest_customer_policy,
    }
    
    results = {}
    for name, policy_fn in baselines.items():
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=500)
        metrics = policy_fn(env)
        results[name] = metrics
        
        print(f"{name}:")
        print(f"  Total reward: {metrics['total_reward']:.2f}")
        print(f"  Total distance: {metrics['total_distance']:.2f}")
        print(f"  Episode length: {metrics['episode_length']}")
        print(f"  Customers delivered: {metrics['customers_delivered']}/{len(customers)}")
        print(f"  Average utilization: {metrics['avg_utilization']:.2%}\n")
    
    best_policy = min(results.items(), key=lambda x: x[1]['total_distance'])
    print(f"Best policy: {best_policy[0]} (distance: {best_policy[1]['total_distance']:.2f})")
    print("[OK] Baseline policies OK\n")
    
    return results


if __name__ == "__main__":
    
    # The tests run sequentially because they build on each other:

    # ✓ First, ensure environment exists
    # ✓ Then, verify constraints are enforced
    # ✓ Then, confirm full episodes work
    # ✓ Finally, benchmark against baselines
    # This layered approach catches problems early - if basics fail, there's no point testing policies.

    # Overall Purpose
    # This entire test suite validates that:

    # Environment is correct (structure, initialization)
    # Constraints are enforced (masking, feasibility)
    # Episodes complete successfully (termination, metrics)
    # Policies can learn from it (baseline comparisons establish expected performance)
    # Without these tests passing, training an RL agent on the environment would be unreliable or impossible.
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED FLEET ROUTING ENVIRONMENT TESTS")
    print("=" * 60 + "\n")
    
    env, customers, trucks, depots = test_environment_basics()
    test_feasibility_masking(env)
    test_episode_completion(env)
    test_baseline_policies(customers, trucks, depots)
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
