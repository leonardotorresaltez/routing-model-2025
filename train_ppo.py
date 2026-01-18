"""
Training script for fleet routing policy using PPO (Proximal Policy Optimization).

Uses stable-baselines3 to train a policy on simplified fleet routing environment.
"""

import os
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines import greedy_nearest_customer_policy
from env.data_generation import generate_problem_instance
from env.routing_env_simple import SimpleFleetRoutingEnv
from env.sb3_wrapper import FleetRoutingSB3Wrapper
from model.graph_converter import (visualize_routing_solution,
                                   visualize_routing_step)
from model.graph_pointer_policy import SimpleGraphPointerPolicy


class GraphVisualizationCallback(BaseCallback):
    """
    Custom callback to visualize routing graphs during training.
    
    Periodically saves graph visualizations showing the evolution of truck
    and customer positions throughout the training process.
    """
    
    def __init__(
        self,
        num_trucks: int,
        num_customers: int,
        viz_freq: int = 5000,
        save_dir: str = "debug_graphs"
    ):
        """
        Args:
            num_trucks: Number of trucks
            num_customers: Number of customers
            viz_freq: Visualization frequency (every N timesteps)
            save_dir: Directory to save visualizations
        """
        super().__init__()
        self.num_trucks = num_trucks
        self.num_customers = num_customers
        self.viz_freq = viz_freq
        self.save_dir = save_dir
        self.step_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.viz_freq == 0:
            try:
                wrapped_env = self.training_env.envs[0]
                base_env = wrapped_env.env
                obs_dict = base_env._get_observation()
                
                flat_obs = wrapped_env._flatten_observation(obs_dict)
                
                depots = [depot.location() for depot in base_env.depots]
                
                truck_routes = {}
                for truck_id, truck_state in enumerate(base_env.truck_states):
                    truck_routes[truck_id] = truck_state.visited_customers
                
                save_path = os.path.join(
                    self.save_dir,
                    f"graph_step_{self.num_timesteps:06d}.png"
                )
                os.makedirs(self.save_dir, exist_ok=True)
                
                visualize_routing_solution(
                    flat_obs,
                    self.num_trucks,
                    self.num_customers,
                    depots=depots,
                    truck_routes=truck_routes,
                    step=self.num_timesteps,
                    title_suffix=f"Training timestep {self.num_timesteps}",
                    save_path=save_path
                )
                print(f"Graph visualization saved at timestep {self.num_timesteps}")
            except Exception as e:
                print(f"Warning: Could not visualize graph at timestep {self.num_timesteps}: {e}")
        
        return True


def make_env(customers, trucks, depots, max_steps=500):
    """Create a wrapped environment factory."""
    def _init():
        env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=max_steps)
        # print(env.observation_space.spaces['feasability_mask'])
        # print(env.observation_space['feasibility_mask'])
        env = FleetRoutingSB3Wrapper(env) # - Wraps the environment to make it compatible with Stable Baselines3 (a RL library). The wrapper adapts the custom environment interface to follow the Gym/Gymnasium standard that Stable Baselines3 expects.
        return env
    return _init


def evaluate_policy(
    policy,
    env,
    num_episodes=10,
    visualize: bool = False,
    num_trucks: int = 4,
    num_customers: int = 15,
    viz_episodes: int = 2
) -> Dict[str, float]:
    """
    Evaluate a trained policy on the environment.
    
    Args:
        policy: trained PPO model
        env: environment for evaluation
        num_episodes: number of episodes to evaluate
        visualize: whether to visualize graphs during evaluation
        num_trucks: number of trucks (for visualization)
        num_customers: number of customers (for visualization)
        viz_episodes: number of episodes to visualize (if visualize=True)
    
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    total_distances = []
    episode_lengths = []
    
    for episode_id in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        should_viz = visualize and episode_id < viz_episodes
        
        while not done:
            if should_viz:
                visualize_routing_step(
                    obs,
                    num_trucks,
                    num_customers,
                    step=steps,
                    title_suffix=f"Evaluation Episode {episode_id}",
                    save_dir="eval_graphs"
                )
            
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



# 1. Collect Experience (64 timesteps = batch_size):
#    └─ Agent takes actions in environment
#    └─ Gets rewards: distance_cost (0.75) + utilization_bonus (0.25)
#    └─ Records: observation, action, reward, value estimate

# 2. Compute Advantages (10 times = n_epochs):
#    └─ Compare: actual_return vs predicted_value
#    └─ Advantage = "was this action better/worse than expected?"
   
# 3. Update Network (3 losses combined):
#    └─ Policy Loss: increase prob of actions with positive advantage
#    └─ Value Loss: make value estimates more accurate (vf_coef=0.5)
#    └─ Entropy Loss: maintain exploration (ent_coef=0.0, disabled)

# 4. Gradient Descent:
#    └─ Adjust weights using backpropagation
#    └─ Clip updates (clip_range=0.2) to prevent drastic changes
#    └─ Apply max gradient norm (max_grad_norm=0.5) for stability
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
    print(f"  Total customer volume: {sum(c.volume for c in customers):.0f} X")
    print(f"  Total truck capacity: {sum(t.max_capacity for t in trucks):.0f} X")
    print(f"  Capacity ratio: {sum(c.volume for c in customers) / sum(t.max_capacity for t in trucks):.2%}\n")
    
    env = DummyVecEnv([make_env(customers, trucks, depots)]) #  create a vectorized environment for parallel training with Stable Baselines3's PPO algorithm.
    
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

    
    policy_kwargs = dict(
        num_trucks=num_trucks,
        num_customers=num_customers,
        hidden_dim=64
    )
    
    # About vf_coef:
    # -----------
    # Policy Loss: Updates the actor (decides which actions to take)
    # Value Loss: Updates the critic (estimates expected cumulative reward)
    #     Used to compute advantage estimates for policy updates
    #     Reduces training variance
    #     Helps stabilize learning
    
        
    # Collect Experience:
    # --------------------------
    # Each timestep, the agent calls the policy with an observation (truck positions, customer locations, truck capacities, etc.)
    # The policy's feature extractor (your SimpleGraphFeaturesExtractor in model/graph_pointer_policy.py:157) processes this through the Graph Pointer Network
    # Two outputs emerge from the extracted features:
    # Policy head: produces action probabilities (which customer to assign to which truck)
    # Value head: predicts expected future reward

    # Execute Action & Get Reward:
    # --------------------------
    # The agent picks an action based on the policy probabilities
    # Environment (env/routing_env_simple.py:129) returns a reward based on:
    # Distance cost (0.75 weight)
    # Utilization efficiency (0.25 weight)

    # Update the Network (This is where the losses matter):
    # --------------------------
    # Policy Loss: Measures how well the policy's action choices led to good rewards. PPO adjusts the policy to increase probability of actions that had positive "advantage" (actual return vs predicted value)
    # Value Loss: Measures prediction error—how far the predicted value was from the actual return. PPO updates the value function to predict better next time
    model = PPO(
        policy=SimpleGraphPointerPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5, # vf_coef=0.5, the value function loss contributes equally to policy loss in the optimization. 

        max_grad_norm=0.5,
        verbose=1,
        device="cpu"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=output_dir,
        name_prefix="ppo_routing"
    )
    
    viz_callback = GraphVisualizationCallback(
        num_trucks=num_trucks,
        num_customers=num_customers,
        viz_freq=10000,
        save_dir=os.path.join(output_dir, "training_graphs")
    )
    
    print("Training...")
    # Stable Baselines3's PPO.learn() expects a vectorized environment. Handled by DummyVecEnv().
    # It simplifies the training loop—you don't manually call env.step() in a loop; the algorithm handles it internally.
    model.learn(
        total_timesteps=num_timesteps,
        callback=[checkpoint_callback, viz_callback],
        progress_bar=False
    )
    
    model.save(os.path.join(output_dir, "ppo_routing_final"))
    print("\nTraining completed. Model saved.")
    
    print("\n" + "=" * 60)
    print("POLICY EVALUATION")
    print("=" * 60)
    
    eval_env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=500)
    eval_env = FleetRoutingSB3Wrapper(eval_env)
    
    metrics = evaluate_policy(
        model,
        eval_env,
        num_episodes=20,
        visualize=True,
        num_trucks=num_trucks,
        num_customers=num_customers,
        viz_episodes=2
    )
    
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
