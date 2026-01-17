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
├── tests/                           # Test suite
│   ├── test_graph_pointer_integration.py  # GPN architecture tests
│   ├── test_environment.py                # Environment tests
│   ├── final_integration_test.py          # System integration tests
│   ├── run_all_tests.py                   # Test orchestrator
│   └── README.md                          # Test documentation
├── model/                           # Neural network policies
│   ├── graph_converter.py          # Observation → graph conversion
│   └── graph_pointer_policy.py     # Graph Pointer Network policy
├── README.md                        # Full problem specification
├── README_SIMPLIFIED.md             # Simplified specification
└── GRAPH_POINTER_IMPLEMENTATION.md  # GPN architecture details
```

## Quick Start

### 1. Test the Environment

Run the complete test suite to verify everything is working:

```bash
python -m tests.run_all_tests
```

This will run:
- **Graph Pointer Network integration tests** (9 tests): Graph conversion, feature extraction, policy
- **Environment tests** (4 tests): Basic functionality, feasibility masking, episode completion, baselines
- **Final integration tests** (6 tests): End-to-end system validation

See `tests/README.md` for detailed test documentation and individual test execution.

### 2. Train a PPO Policy

Train a policy with default parameters (20 customers, 5 trucks, 100,000 timesteps):

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

## Policy Architecture

### Graph Pointer Network Policy

The system uses a **Graph Pointer Network** (GPN) policy that:

1. **Converts observations to graphs**: Flat 80-dimensional observations are converted to explicit graph structures with:
   - Node features: Truck and customer positions [T+N, 2]
   - Adjacency matrix: k-nearest neighbors spatial connectivity [T+N, T+N]

2. **Processes through attention**: The GraphPointerNetwork uses multi-head attention to score nodes

3. **Combines with MLP features**: Graph-aware embeddings are fused with traditional MLP features

4. **Outputs actions**: Produces action logits for each truck's customer selection

See `GRAPH_POINTER_IMPLEMENTATION.md` for architecture details.

### Baseline Policies

For comparison, `baselines.py` implements three simple policies:

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
- Verify feasibility masking is working: `python -m tests.test_environment`
- Ensure Graph Pointer Network is active: check output contains "graph" mentions
- Try different hyperparameter combinations

## Next Steps

1. **Graph Pointer Tuning**: Adjust `k_neighbors` and `hidden_dim` in Graph Pointer Network for better performance
2. **Longer training**: Run with `--timesteps 500000` or more for better convergence
3. **Custom reward tuning**: Adjust weights in `routing_env_simple.py` to balance distance vs. utilization
4. **Curriculum learning**: Generate progressively harder problem instances to train on
5. **Multi-problem training**: Train on diverse problem sizes and depot configurations
6. **Policy transfer**: Fine-tune pre-trained models on new problem instances
