# Fleet Routing Training Guide

## Overview

This guide explains how to use the simplified fleet routing environment and train policies using reinforcement learning.

## Environment Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `gymnasium` - RL environment API
- `networkx` - Graph utilities
- `numpy` - Numerical computing
- `stable-baselines3` - PPO and other RL algorithms
- `torch` - Neural network backend

### File Structure

```
routing-model-2025/
├── env/
│   ├── routing_env_simple.py      # Main environment class
│   ├── types_simple.py             # Data structures (Customer, Truck, Depot)
│   ├── utils_simple.py             # Helper functions (distance, feasibility)
│   ├── data_generation.py          # Generate random problem instances
│   └── sb3_wrapper.py              # Stable-Baselines3 wrapper for the environment
├── baselines.py                     # Greedy baseline policies
├── train_ppo.py                     # PPO training script
├── test_environment.py              # Environment validation tests
├── README.md                        # Full problem specification
└── README_SIMPLIFIED.md             # Simplified problem specification
```

## Quick Start

### 1. Test the Environment

Run the environment tests to verify everything is working:

```bash
python test_environment.py
```

This will:
- Test basic environment functionality
- Validate feasibility masking
- Run greedy baseline policies
- Display performance metrics

### 2. Train a PPO Policy

Train a policy with default parameters (15 customers, 4 trucks, 10,000 timesteps):

```bash
python train_ppo.py
```

**Key Parameters:**

- `--timesteps`: Total training timesteps (default: 100,000)
- `--customers`: Number of customers in problem (default: 20)
- `--trucks`: Number of trucks (default: 5)
- `--depots`: Number of depots (default: 2)
- `--lr`: Learning rate (default: 0.0003)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: PPO update epochs (default: 10)
- `--output-dir`: Directory to save models (default: `models`)

**Example:**

```bash
python train_ppo.py --timesteps 100000 --customers 25 --trucks 8 --depots 3 --lr 1e-3
```

### 3. Output and Evaluation

The training script will:
1. Generate a random problem instance
2. Train a PPO policy on this instance
3. Save the trained model to `models/ppo_routing_final.zip`
4. Evaluate the trained policy on 20 evaluation episodes
5. Compare performance against greedy baseline
6. Display metrics:
   - Mean episode reward
   - Mean total distance traveled
   - Mean truck utilization
   - Comparison against baseline

## Understanding the Environment

### Observation Space

The environment provides observations as a flattened vector containing:

- **Truck states**: Position (x, y), current load, max capacity, utilization
- **Customer states**: Location (x, y), delivery weight, visitation status
- **Feasibility masks**: Which actions are valid for each truck

Total observation size: `5 * num_trucks + 4 * num_customers`

### Action Space

**MultiDiscrete**: Each truck chooses one action per step.

- Valid actions: [0, num_customers] where each value is a customer ID
- Special action: `num_customers` = return to home depot
- Action masking ensures only feasible actions are available

### Reward Function

```
R(t) = 0.75 * r_routing + 0.25 * r_efficiency
```

**r_routing**: Negative distance cost (minimize travel)
```
r_routing = -distance_traveled
```

**r_efficiency**: Truck utilization bonus
```
+5.0 if utilization > 0.8
+2.0 if 0.5 < utilization <= 0.8
-1.0 if utilization < 0.3
```

### Episode Termination

- **Success**: All customers delivered
- **Failure**: Max steps exceeded (truncation)

## Baseline Policies

The `baselines.py` module implements three simple policies for comparison:

1. **Greedy Nearest Customer**: Each truck picks the nearest unvisited customer it can serve
2. **Nearest Depot First**: Trucks return to depot after reaching 70% utilization
3. **Furthest Customer**: Exploratory policy that picks farthest customers (spreads coverage)

## Advanced Usage

### Custom Problem Generation

```python
from env.data_generation import generate_problem_instance
from env.routing_env_simple import SimpleFleetRoutingEnv

# Generate a custom problem
customers, trucks, depots = generate_problem_instance(
    num_customers=30,
    num_trucks=10,
    num_depots=3,
    truck_capacity_range=(1000, 2000),
    customer_weight_range=(50, 200),
    seed=42
)

# Create environment
env = SimpleFleetRoutingEnv(customers, trucks, depots, max_steps=1000)
```

### Curriculum Learning

```python
from env.data_generation import generate_curriculum_instances

# Generate progressively harder problem instances
instances = generate_curriculum_instances(
    difficulty_levels=5,
    base_customers=10,
    base_trucks=3
)

# Train on easy problems first, then harder ones
```

### Evaluate Existing Model

```python
from stable_baselines3 import PPO
from env.routing_env_simple import SimpleFleetRoutingEnv
from env.sb3_wrapper import FleetRoutingSB3Wrapper

# Load trained model
model = PPO.load("models/ppo_routing_final")

# Create environment
env = SimpleFleetRoutingEnv(customers, trucks, depots)
env = FleetRoutingSB3Wrapper(env)

# Run episodes
for episode in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"Episode {episode}: distance={info['total_distance']:.2f}")
```

## Hyperparameter Tuning

The following hyperparameters affect training performance:

**Environment:**
- `reward_routing_weight` (0.75): How much distance cost matters
- `reward_efficiency_weight` (0.25): How much utilization bonus matters
- Utilization thresholds: (0.8, 0.5, 0.3) for bonus/penalty boundaries

**PPO Algorithm:**
- `learning_rate` (3e-4): Gradient step size
- `batch_size` (64): Samples per gradient step
- `n_epochs` (10): Policy update iterations per rollout
- `gamma` (0.99): Discount factor for returns
- `clip_range` (0.2): PPO clipping range

Start with defaults and adjust if convergence is slow or training is unstable.

## Expected Performance

For a 15-customer, 4-truck, 2-depot problem:

- **Greedy baseline**: ~800-900 units of total distance, 8 steps
- **PPO (10k steps training)**: ~1000-2000 units (not yet converged)
- **PPO (100k+ steps)**: Should approach or beat baseline

Note: Longer training, larger batch sizes, and higher learning rates may improve convergence.

## Troubleshooting

**Training is very slow:**
- Reduce `num_customers` or `num_trucks` for faster episodes
- Use larger `batch_size` (e.g., 128 or 256)
- Try higher learning rate (e.g., 1e-3)

**Validation error "could not broadcast...":**
- Ensure the observation space size formula in `sb3_wrapper.py` matches actual observations
- Run `debug_obs_size.py` if you customize problem parameters

**Model not improving over baseline:**
- Train for more timesteps (100k+)
- Verify feasibility masking is working (check `test_environment.py`)
- Try different hyperparameter combinations

## Next Steps

1. **Longer training**: Run with `--timesteps 500000` or more
2. **Custom reward tuning**: Adjust weights in `routing_env_simple.py`
3. **Architecture improvements**: Replace `MlpPolicy` with custom networks
4. **Graph Neural Networks**: Implement GNN-based policy for relational reasoning
5. **Multi-problem training**: Train on curriculum of varying difficulty levels
