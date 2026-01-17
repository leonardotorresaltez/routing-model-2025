"""
Training script for fleet routing policy using PPO (Proximal Policy Optimization).

Uses stable-baselines3 to train a policy on simplified fleet routing environment.
"""

import os
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines import greedy_nearest_customer_policy
from env.data_generation import generate_problem_instance
from env.routing_env_simple import SimpleFleetRoutingEnv
from env.sb3_wrapper import FleetRoutingSB3Wrapper


def make_env(customers, trucks, depots, max_steps=500):
    """Create a wrapped environment factory."""
    def _init():
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=max_steps)
        env = FleetRoutingSB3Wrapper(env)
        return env
    return _init


def evaluate_policy(policy, env, num_episodes=10) -> Dict[str, float]:
    """
    Evaluate a trained policy on the environment.
    
    Args:
        policy: trained PPO model
        env: environment for evaluation
        num_episodes: number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    total_distances = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            action, _states = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_distances.append(info.get("total_distance", 0.0))
        episode_lengths.append(steps)
    
    return {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_distance": float(np.mean(total_distances)),
        "std_distance": float(np.std(total_distances)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }


def train(
    num_timesteps: int = 100000,
    num_customers: int = 20,
    num_trucks: int = 5,
    num_depots: int = 2,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    output_dir: str = "models"
):
    """
    Train PPO policy on fleet routing environment.
    
    Args:
        num_timesteps: total training timesteps
        num_customers: number of customers in problem
        num_trucks: number of trucks
        num_depots: number of depots
        learning_rate: PPO learning rate
        batch_size: training batch size
        n_epochs: PPO update epochs
        output_dir: directory to save models
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("FLEET ROUTING PPO TRAINING")
    print("=" * 60)
    print(f"Customers: {num_customers}")
    print(f"Trucks: {num_trucks}")
    print(f"Depots: {num_depots}")
    print(f"Training timesteps: {num_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"N epochs: {n_epochs}\n")
    
    customers, trucks, depots = generate_problem_instance(
        num_customers=num_customers,
        num_trucks=num_trucks,
        num_depots=num_depots,
        seed=42
    )
    
    print(f"Generated problem instance:")
    print(f"  Total customer weight: {sum(c.weight for c in customers):.0f} kg")
    print(f"  Total truck capacity: {sum(t.max_capacity for t in trucks):.0f} kg")
    print(f"  Capacity ratio: {sum(c.weight for c in customers) / sum(t.max_capacity for t in trucks):.2%}\n")
    
    env = DummyVecEnv([make_env(customers, trucks, depots)])
    
    # AGENT  
    
    # What the Agent Does - Two Phases
    #     PHASE 1: TRAINING (Learning)
    #         During agent.learn():

    #         Collects experience from environment

    #         Interacts with the environment thousands of times
    #         Stores (observation, action, reward) tuples
    #         Updates the policy (improves neural network weights)

    #         Analyzes collected experience
    #         Adjusts the policy's neural network weights
    #         Goal: make better decisions next time
    #         Tracks metrics

    #         Monitors loss (how wrong predictions are)
    #         Tracks entropy (exploration vs exploitation balance)
    #         Watches gradient norms (learning speed)
            
    #     PHASE 2: DEPLOYMENT (Decision Making)
    #         During evaluation or real use:

    #         Makes decisions

    #         Receives observation from environment (truck positions, customer locations)
    #         Uses the learned policy to choose best action
    #         Decides: which customer should each truck visit next?
    #         Uses the policy (neural network)

    #         The policy has learned patterns from training
    #         Takes observation as input → outputs action probabilities
    #         Selects the action with highest probability

    
    model = PPO( # AGENT!   
        policy="MlpPolicy",  # <-- The POLICY
        env=env, # The agent interacts with the environment
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix="ppo_routing"
    )
    
    print("Training...")
    model.learn(
        total_timesteps=num_timesteps,
        callback=checkpoint_callback,
        progress_bar=False
    )
    
    model.save(os.path.join(output_dir, "ppo_routing_final"))
    print("\nTraining completed. Model saved.")
    
    print("\n" + "=" * 60)
    print("POLICY EVALUATION")
    print("=" * 60)
    
    eval_env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=500)
    eval_env = FleetRoutingSB3Wrapper(eval_env)
    
    metrics = evaluate_policy(model, eval_env, num_episodes=20)
    
    print(f"Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"Mean distance: {metrics['mean_distance']:.2f} +/- {metrics['std_distance']:.2f}")
    print(f"Mean episode length: {metrics['mean_length']:.1f} +/- {metrics['std_length']:.1f}")
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    
    baseline_env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=500)
    baseline_metrics = greedy_nearest_customer_policy(baseline_env)
    
    print(f"Greedy Nearest Baseline:")
    print(f"  Total distance: {baseline_metrics['total_distance']:.2f}")
    print(f"  Average utilization: {baseline_metrics['avg_utilization']:.2%}")
    print(f"  Episode length: {baseline_metrics['episode_length']}")
    
    print(f"\nPPO Policy vs Baseline:")
    improvement = (baseline_metrics['total_distance'] - metrics['mean_distance']) / baseline_metrics['total_distance']
    print(f"  Distance improvement: {improvement:.2%}")
    
    if improvement > 0:
        print(f"  PPO is {improvement:.2%} better than baseline")
    else:
        print(f"  Baseline is {-improvement:.2%} better than PPO")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO policy for fleet routing")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--customers", type=int, default=20, help="Number of customers")
    parser.add_argument("--trucks", type=int, default=5, help="Number of trucks")
    parser.add_argument("--depots", type=int, default=2, help="Number of depots")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    
    args = parser.parse_args()
    
    train(
        num_timesteps=args.timesteps,
        num_customers=args.customers,
        num_trucks=args.trucks,
        num_depots=args.depots,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        output_dir=args.output_dir
    )
