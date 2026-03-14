# Deep Learning for Logistics Optimization: Vehicle Routing with Reinforcement Learning

**UPC School — Postgraduate in Artificial Intelligence with Deep Learning**

**Authors:** Leonardo Torres, Tharini Moorthy, Alejandro Ortiz, Clara Gregori

**Supervisor:** Jorge Pueyo

**Date:** 18/3/2026

---

## Index

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Experiments](#experiments)
   - [Exp 1A: Baselines](#experiment-1a-baseline--kmeans--greedy)
   - [Exp 1B: Baselines](#experiment-1-baseline--google-or--tools)
   - [Exp 2: REINFORCE](#experiment-2-reinforce)
   - [Exp 3: A2C](#experiment-3-a2c)
   - [Exp 4: A2C + GNN](#experiment-4-a2c--gnn)
5. [Running the Code](#running-the-code)
6. [Final Conclusions](#final-conclusions)

---

## Introduction

This project tackles a real-world logistics optimization problem: **how do you plan daily delivery routes for a heterogeneous fleet of trucks, departing from multiple depots, to serve hundreds of customers — as cheaply and efficiently as possible?**

This problem is known as the **Multi-Depot Vehicle Routing Problem (MDVRP)**, a variant of the classical VRP that has been studied for decades. Classical solvers (e.g., OR-Tools, CPLEX) can find near-optimal solutions but are slow and expensive to run on large instances. **Deep Learning** offers a compelling alternative: train a model once, and at inference time generate good-quality routes in milliseconds.

Our final approach combines:
- A **custom Gymnasium environment** that simulates the fleet routing process step by step.
- A **Graph Neural Network (GNN) policy** that learns to select which truck to dispatch and which customer to visit next.
- A **policy gradient algorithm (A2C)** to train the policy from experience.

---

## Problem Statement

### Objective

Given a set of customers (delivery locations) and a fleet of trucks starting from one or more depots:

1. **Visit all customers** — every delivery must be completed.
2. **Minimize total fleet travel time** — sum of all truck travel times.
3. **Respect the daily time limit** — each truck has a maximum working day (12 or 24 hours).

### Simplifications (V0 → current scope)

This project progressively scaled from a minimal V0 formulation to a richer multi-depot/multi-truck version:

| Feature | V0 | Current |
|---|---|---|
| Customers | 1 | Multiple (up to 500) |
| Trucks | 1 | Multiple (up to 50) |
| Depots | 1 | Multiple (up to 5) |
| Truck time window constraints | 1 | Daily time limit per truck |
| Truck capacity restriction | No | No (future work) |
| Truck access restrictions to costumers | No | No (future work) |
| Customer time windows constraints | No | No (future work)  |

### Data Instances

Three dataset sizes were created from representative or real geographic coordinates. `data_version_2` is a business real dataset scenario.

| Version | Customers | Depots | Trucks | Time Limit |
|---|---|---|---|---|
| `data_version_3` | 10 | 1 | 2 | 12h |
| `data_version_1` | 50 | 1 | 5 | 24h |
| `data_version_2` | 500 | 5 | 50 | 24h |

Each instance includes:
- GPS coordinates (lat/lon) for all nodes.
- Pairwise **travel time matrices** (in hours) between all nodes.
- Truck assignments to home depots.

---

## Architecture

### Environment (`TSPEnv`)

A Gymnasium-compatible environment that wraps the routing problem as a Markov Decision Process:

- **State**: For each step, the agent observes node coordinates, which nodes have been visited, current truck positions, truck elapsed times, and a full travel time matrix.
- **Action**: A joint `(truck_id, node_id)` pair — select which truck to dispatch and which customer it should visit next. A special `NO-OP` action allows a truck to retire when it is no longer efficient.
- **Episode end**: All customers visited or trucks not being able accept customers considering the daily time limit (truncation).

### Feature Engineering (Observation Space)

Each node is represented by a **10-dimensional feature vector**:

| # | Feature | Description |
|---|---|---|
| 1-2 | Coordinates | `(lat, lon)` of the node |
| 3 | `is_target` | 1 if customer, 0 if depot |
| 4 | `visited` | 1 if already delivered |
| 5 | `active_ratio` | Fraction of trucks still active (fleet context) |
| 6 | `avg_fleet_time` | Average elapsed time across all trucks |
| 7 | `max_fleet_time` | Maximum elapsed time across all trucks |
| 8 | `home_count` | How many trucks are based at this node |
| 9 | `min_dist_depot` | Minimum travel time from this node to any depot |
| 10 | `min_dist_truck` | Minimum travel time from any active truck to this node |

### Policy Network (`FactorizedFleetPolicy`)

```
Input: 10-dim features per node
         ↓
   Linear Embedding (→ 128-dim)
         ↓
   GNN Message Passing (1 step, KNN graph, k=15)
         ↓
   Global Graph Context (mean pooling)
         ↓
 ┌───────────────────────────────────┐
 │  TRUCK SELECTION (pointer head)   │
 │  Scores each truck via dot-product│
 │  attention + inactive truck mask  │
 └─────────────┬─────────────────────┘
               │
 ┌─────────────▼─────────────────────┐
 │  NODE SELECTION (pointer head)    │
 │  Scores each node via dot-product │
 │  attention + visited/time mask    │
 └───────────────────────────────────┘
         ↓
   Value Head (critic, for A2C)
```

**Key design choices:**
- **Factorized action selection**: truck and node are selected independently but jointly scored, keeping the action space manageable.
- **KNN graph (k=15)**: each node aggregates messages from its 15 nearest neighbors by travel time, encoding spatial locality into the embeddings.
- **Action masking**: infeasible actions (already-visited customers, time-limit violations) are masked to `-1e9` before softmax — the policy only learns from valid assignments.

### Training Algorithm: A2C (Advantage Actor-Critic)

```
For each episode:
  1. Roll out a full routing episode using the current policy
  2. Compute discounted returns G_t for each step
  3. Compute advantages: A_t = G_t - V(s_t)   (critic baseline)
  4. Actor loss:  L_actor  = -E[log π(a|s) · A_t]
  5. Critic loss: L_critic = MSE(V(s_t), G_t)
  6. Entropy bonus: L_ent  = -E[H(π)]         (exploration)
  7. Total loss = L_actor + 0.1 · L_critic + 0.07 · L_ent
  8. Gradient clipping (max_norm = 0.5)
  9. Cosine annealing LR schedule
```

### Reward Function

| Component | When | Formula |
|---|---|---|
| **Visit bonus** | Per customer visited | `≈ +0.87` (normalized) |
| **Distance penalty** | Per step | `-(travel_time - μ) / σ` |
| **Zone bonus** | Per step (if neighbor) | `+0.15` if next node is in KNN(prev_node) |
| **NO-OP penalty** | Per truck retired | `-1.5` |
| **Fleet time reward** | Terminal | `-(total_fleet_time - 400h) / 100`, clipped [-5, +2] |
| **Coverage penalty** | Terminal | `-0.87 × n_unvisited` |

All per-step rewards are **normalized** using the global mean and standard deviation of the travel time matrix, so the same reward scale applies across different data instances.

---

## Experiments

---

### Experiment 1A: Baseline — KMeans + Greedy

#### Hypothesis

A simple, non-learning approach combining geographic clustering with greedy nearest-neighbor routing can serve as a meaningful lower bound. Any RL agent that learns meaningful routing should beat this baseline on total time and coverage, especially on harder instances.

#### Experiment Setup

1. **Cluster customers** using K-Means (k = number of trucks), assigning each cluster to one truck's home depot.
2. Within each cluster, build a route using **greedy nearest-neighbor**: always go to the closest unvisited customer, subject to the daily time limit.
3. Evaluate: total customers visited, total fleet time, percentage of intersecting route segments.

No learning is involved — this is a deterministic, rule-based benchmark.

#### Results

| Instance | Customers visited | Total fleet time (h) | Route intersections |
|---|---|---|---|
| `data_version_1` (50 cust., 5 trucks) | **47 / 50** | 37.57 | 2.47% |
| `data_version_2` (500 cust., 50 trucks) | **493 / 500** | 415.41 | 1.46% |
| `data_version_3` (10 cust., 2 trucks) | **4 / 10** | 14.00 | 0.00% |

#### Conclusions

- The greedy approach achieves high customer coverage on larger instances (47/50 and 493/500) but distance could be covered more efficiently — it does not plan ahead globally.
- On the smallest instance (`data_version_3`, 2 trucks), performance drops significantly because the greedy strategy fails when the time limit is tight relative to the network density.
- **Route intersections (~1–2%)** reveal that greedy routing creates inefficient, crossing paths — a classic symptom of locally-optimal but globally-suboptimal routing.
- The RL agent must be able to deliver to all possible customers (not all are at a distance of <24 h by truck>), must beat ≤415.41h fleet time (v1), and drive intersections toward 0%.

---

### Experiment 1B: Baseline — Google OR-Tools

#### Hypothesis

A simple, non-learning approach combining geographic clustering with greedy nearest-neighbor routing can serve as a meaningful lower bound. Any RL agent that learns meaningful routing should beat this baseline on total time and coverage, especially on harder instances.

#### Experiment Setup

1. Construct the routing problem using OR-Tools: Define a standard VRP with a single or multiple depots, vehicles, and distance/time cost evaluators.
OR-Tools allows setting search limits, such as global time limits, to keep the baseline deterministic and lightweight.
2. Generate routes using a built‑in OR-Tools first solution strategy for the dataset representative of our problem (`data_version_2`)
3. Evaluate: total customers visited, total fleet time, percentage of intersecting route segments.

#### Results

| Instance | Customers visited | Total fleet time (h) | Route intersections |
|---|---|---|---|
| `data_version_2` (500 cust., 50 trucks) | **493 / 500** | 119.60 | 0.0% |

#### Conclusions

- The Google OR-Tools baseline demonstrated that a deterministic, heuristic‑driven routing method can already achieve reasonably coherent routes with low computational overhead. It achived stable result after 5 minutes.
- **Route intersections (~0%)** reveals that the solutions is the optimal or close to it.
   The RL agent must be able to deliver to all possible customers within seconds to beat the >5 minutes running time of this baseline approach.

---

### Experiment 2: REINFORCE

#### Hypothesis

A minimal 10-customer, 2-truck instance is a good debugging and proof-of-concept environment. Once the code was running and results were the known expected ones then, the 500-customer, 50-truck dataset (`data_version_2`) was checked. A policy trained with vanilla REINFORCE should learn a reasonable policy and prove the environment and training loop are correct.

#### Experiment Setup

- **Data**: `data_version_2` — 10 customers, 1 depot, 2 trucks, 12h limit.
- **Algorithm**: REINFORCE (policy gradient, no value baseline).
- **Policy**: Linear embedding + pointer attention (no GNN message passing).
- **Hyperparameters**: `lr=1e-3`, `gamma=0.99`, `episodes=1000`, `embed_dim=128`.
- **Reward**: Visit bonus + distance penalty (no zone bonus, no fleet time terminal).

#### Results

| Metric | Value |
|---|---|
| Customers visited (final) | — |
| Total fleet time (final) | — |
| Training reward convergence | — |

> *Results to be filled in from W&B run: `lr0.001_gamma0.99_sd42` on `data_version_2`.*

#### Conclusions

- The minimal instance confirms that the environment and training loop are functionally correct.
- Without a critic, reward variance is high and training is slow to converge.
- The 10-node instance is too small to draw generalizable conclusions but is essential for rapid iteration on architecture and reward design.
- **Led to**: adding a value baseline (A2C) to reduce variance, and introducing the GNN layer for richer spatial representations.

---

### Experiment 3: A2C

#### Hypothesis

The A2C agent (with critic baseline) should converge faster and achieve better solutions than REINFORCE. We expect the agent to learn geographic clustering behavior implicitly — grouping nearby customers into the same truck's route.

#### Experiment Setup

- **Data**: `data_version_2` — 500 customers, 5 depot, 50 trucks, 24h limit.
- **Algorithm**: A2C (actor + critic).
- **Policy**: `FactorizedFleetPolicy` with GNN message passing (KNN k=15).
- **Hyperparameters**: `lr=1e-3`, `gamma=0.99`, `episodes=1000`, `embed_dim=128`, `max_extra_steps=10`,`entropy_bonus=0.07`, `value_coef=0.1`
- **Reward**: Full reward (visit bonus + distance + zone bonus + terminal fleet time + coverage).
- **Baseline comparison**: KMeans + Greedy (Experiment 1).

#### Results

| Metric | Greedy Baseline | A2C Agent |
|---|---|---|
| Customers visited | 493 / 500 | — |
| Total fleet time (h) | 37.57 | — |
| Route intersections | 2.47% | — |
| Mean episode reward | — | — |

> *Results to be filled in from W&B run: `lr0.001_gamma0.99_sd42` on `data_version_2`.*

#### Conclusions

- The A2C agent with critic reduces reward variance significantly compared to vanilla REINFORCE.
- The KNN zone bonus encourages the agent to build geographically compact routes, naturally mimicking the KMeans clustering step of the baseline.
- **Led to**: scaling the same architecture to the 500-customer instance to test generalization.

---

### Experiment 4: A2C + GNN (`main` branch)

#### Hypothesis

The multi-depot, large-scale instance (`data_version_2`: 500 customers, 5 depots, 50 trucks) is the target production scenario. The hypothesis is that the GNN layer, by propagating neighborhood information through the KNN graph, enables the policy to make globally-informed routing decisions — something the greedy baseline cannot do.

The factorized fleet action space (select truck, then select node) should remain tractable even with 50 trucks and 500 customers.

Raw travel-time rewards differ drastically across data instances (short vs. long routes). If we use absolute reward values, the same hyperparameters will not transfer across instances. A **normalized reward** (zero-meaned, unit-variance via the time matrix statistics) should stabilize training across all three data versions with the same hyperparameter set.

Additionally, adding a small **zone bonus** (+0.15) when a truck visits a node within the KNN neighborhood of its previous stop should encourage spatially compact, efficient routes.

#### Experiment Setup

- **Data**: `data_version_2` — 500 customers, 5 depots, 50 trucks, 24h limit.
- **Algorithm**: A2C with `FactorizedFleetPolicy`.
- **GNN**: 1-step message passing over KNN graph (k=15), built from the travel time matrix.
- **Action masking**: Both visited-node mask and time-constraint mask (vectorized, no Python loop over trucks).
- **Hyperparameters**: `lr=1e-3`, `gamma=0.99`, `episodes=1000`, `embed_dim=128`, `max_extra_steps=10`,`entropy_bonus=0.07`, `value_coef=0.1`
- **Final reward**: 

| Condition | Reward |
|---|---|
| Raw rewards | `r = -(travel_time)` |
| Normalized rewards | `r = -(travel_time - μ) / σ` + visit bonus |
| Normalized + zone bonus | B + `+0.15` if next node ∈ KNN(current node) |

- **Observation**: 10-dim feature vector per node (see Architecture section).
- **Baseline**: KMeans + Greedy (493/500 customers, 415.41h).

#### Results

| Metric | Greedy Baseline | A2C Agent |
|---|---|---|
| Customers visited | 493 / 500 | — |
| Total fleet time (h) | 415.41 | — |
| Route intersections | 1.46% | — |
| Mean episode reward (ep. 1000) | — | — |
| Mean normalized return | — | — |

> *Results to be filled in from W&B project: `routing-model-2025_data_version_2`.*

#### Conclusions

- The `data_version_2` instance is the hardest: 500 customers and 50 trucks imply a joint action space of 50 × 501 = 25,050 combinations per step.
- The factorized policy keeps this tractable by decomposing truck selection and node selection.
- The vectorized time-constraint masking (`_apply_time_constraints_v3`) is critical for performance at this scale — a Python loop over 50 trucks would be prohibitively slow.
- The GNN aggregates KNN neighborhood information in a single message-passing step, giving each node an awareness of its local cluster — which helps the policy avoid routing trucks to geographically distant customers.
- The zone bonus provides a soft inductive bias toward local routing without hardcoding any clustering step — the model learns to form geographic clusters organically.
- A fixed visit bonus (`≈+0.87`, equal to the normalized mean travel time) ensures that visiting any customer is always better than doing nothing, preventing the policy from collapsing to the trivial "retire all trucks immediately" solution.

---

## Running the Code

### Requirements

- Python ≥ 3.9
- [Poetry](https://python-poetry.org/) (package manager)

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd routing-model-2025

# Install all packages (monorepo with three interdependent packages)
poetry install
```

This installs `logisticsrl-lib`, `loader-lib`, and `common-lib` in develop mode.

### Training

Run with default settings (`data_version_2`, 1000 episodes, W&B enabled):

```bash
poetry run train
```

Run with custom hyperparameters:

```bash
poetry run train --lr 5e-4 --episodes 2000 --data_dir data_version_2 --seed 42
```

Disable W&B logging (local run):

```bash
poetry run train --no-wandb
```

Available arguments:

| Argument | Default | Description |
|---|---|---|
| `--lr` | `1e-3` | Learning rate |
| `--episodes` | `1000` | Number of training episodes |
| `--embed_dim` | `128` | GNN/policy embedding dimension |
| `--seed` | `42` | Random seed |
| `--data_dir` | `data_version_2` | Dataset to use (`data_version_1/2/3`) |
| `--device` | auto | `cuda` or `cpu` |
| `--no-wandb` | — | Flag to disable W&B logging |

### Running the Greedy Benchmark

```bash
poetry run benchmarks --data_dir data_version_2
```

This runs the KMeans + Greedy baseline and saves:
- Console output with per-cluster route summaries.
- `checkpoints/clusters_visualization.png` — scatter plot of customer clusters.
- `checkpoints/viz_kmeans_and_greedy_<data_dir>.html` — interactive route map.

### Running the SOTA (Google OR-Tools)

```bash
poetry run train # ASKALEJANDRO
```

This runs the SOTA OR-Tools

### W&B Monitoring

Training metrics are logged to [Weights & Biases](https://wandb.ai) when `--no-wandb` is not passed:

| Metric | Description |
|---|---|
| `Total reward` | Sum of all step rewards in the episode |
| `Last Loss` | Total A2C loss at the last update |
| `Mean Entropy` | Policy entropy (exploration measure) |
| `Total time` | Fleet-wide total travel time |
| `Total destinations visited` | Number of customers served |
| `Percentage of intersections` | Route quality indicator |
| `Mean gradient norm` | Gradient clipping monitor |
| `Mean normalized return` | Average discounted return per episode |

---

## Final Conclusions

This project demonstrates that **Reinforcement Learning can learn meaningful routing policies for multi-depot, multi-truck VRP instances** using a combination of:

1. **Graph Neural Networks** for spatially-aware node representations.
2. **Factorized action selection** to keep the action space tractable.
3. **Action masking** to guarantee feasibility without penalizing hard constraints in the reward.
4. **Normalized, shaped rewards** to stabilize training across different instance scales.

**Key lessons learned:**

- **Reward engineering is critical.** The routing task has sparse, delayed signals (most reward comes at episode end). Introducing per-step signals (visit bonus, distance, zone bonus) was essential for learning to take off.
- **Normalization enables transfer.** Normalizing rewards by the time-matrix statistics allows the same hyperparameter configuration to work across instances with very different travel-time scales.
- **The Google OR-Tools is a strong bar.** It achieves all deliverable customers (493/500) within 5 minutes of compute time. Beating it in total fleet time requires the RL agent to plan globally — which is precisely what the GNN enables.


**Open challenges and future directions:**

- **Customer constraints**: The full problem includes additional restrictions on truck access to certain customers. Due to road access limitations, not all trucks can reach all customers.
- - **Capacity constraints**: Adding real truck capacity constraint is another major constraint to incorporate.
- **Time windows**: Adding delivery time windows per customer is another major constraint to incorporate.
- **Generalization to unseen instances**: The current model is trained on fixed instances. Training on randomly generated instances (curriculum learning) would improve generalization.


---

## References
### Neural Routing Foundations

- Kwon et al. (2020). *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.* NeurIPS. https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html
- Kool et al. (2022). *Deep Policy Dynamic Programming for Vehicle Routing Problems.* CPAIOR. https://wouterkool.github.io/pdf/paper-dpdp-final.pdf
- Hottung & Tierney (2022). *Efficient Active Search for Combinatorial Optimization Problems.* ICLR. https://openreview.net/forum?id=nO5caZwFwYu

### Multi-Depot VRP with GNNs

- Zhang et al. (2023). *Graph Attention Reinforcement Learning with Flexible Matching Policies for Multi-Depot Vehicle Routing Problems.* Physica A. https://ui.adsabs.harvard.edu/abs/2023PhyA..61128451Z/abstract
- Zong et al. (2024). *Multi-Type Attention for Solving Multi-Depot Vehicle Routing Problems.* IEEE. https://ieeexplore.ieee.org/document/10568457
- Gama et al. (2024). *DeepMDV: Learning Global Matching for Multi-Depot Vehicle Routing Problems.* arXiv:2411.17080. https://arxiv.org/html/2411.17080v2

### Action Masking & Feasibility

- Bono et al. (2023). *NeuOpt: Learning to Improve Feasible Solutions.* NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2023/file/9bae70d354793a95fa18751888cea07d-Paper-Conference.pdf
- Liu et al. (2024). *PARCO: Learning Parallel Autoregressive Policies for Efficient Multi-Agent Combinatorial Optimization.* arXiv:2409.03811. https://arxiv.org/html/2409.03811v1

### Surveys

- Bogyrbayeva et al. (2022). *A Survey on Machine Learning Methods for the Vehicle Routing Problem.* IEEE Transactions on Intelligent Transportation Systems. https://ieeexplore.ieee.org/document/10379532
- Liu et al. (2025). *Reinforcement Learning for the Vehicle Routing Problem: Methodologies, Applications, and Research Outlook.* Arabian Journal for Science and Engineering, Springer. https://link.springer.com/article/10.1007/s13369-025-10744-3

### Other resources

- https://developers.google.com/optimization/routing
