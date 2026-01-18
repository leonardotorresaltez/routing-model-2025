# Deep Learning for Fleet Routing Optimization (Simplified Version)
**Title:** Maximize Truck volume Utilization with Deep Learning  
**Date:** 2026-01-17
**Version:** 1.0 (Simplified)

---

## Main Objectives
1. Deliver goods to all clients (no deadline constraints).
2. Minimize total travel distance for the fleet.
3. Maximize volume utilization on each truck.

---

## Problem Statement

Design a system that optimizes routing for a fleet of trucks as a single unit, assigning all customers to trucks such that:
- All customers are visited exactly once
- Each truck carries its delivery assignments and returns to its **home depot** only
- Total fleet distance is minimized
- Truck volume utilization is maximized

**Key Simplifications**:
- ✅ Single objective: minimize distance + maximize utilization
- ✅ No time windows, no driver hour limits
- ✅ Trucks return only to their home depot (no re-pickup from other depots)
- ✅ Clients are homogeneous (no types, no SLA)
- ✅ No packing/pallets (volume-only constraints)

---

## 2. Why This Problem?

- **Real-world relevance**: Pure Vehicle Routing Problem (VRP) with fleet-level optimization
- **Simplified but meaningful**: Focus on core routing + utilization without packing complexity
- **Deep learning advantage**: Neural networks can learn better load-balancing than greedy heuristics
- **Parallel decisions**: All trucks decide simultaneously; the model learns emergent fleet coordination

---

## 3. Environment Definition

### STATE (Multi-Agent Graph)

**Graph Nodes**:
- **Truck Nodes** (T total, one per truck):
  - Current location: (x, y) coordinate
  - Home depot: depot ID where truck must return at day end
  - Current load: total volume of assigned customers (kg)
  - Max capacity: maximum volume truck can carry (kg)
  - Route: list of visited customer IDs

- **Depot Nodes** (D static):
  - Fixed location (x, y)
  - Each truck assigned to exactly one home depot

- **Customer Nodes** (N undelivered customers):
  - Location: (x, y) coordinate
  - volume: customer delivery volume (kg)

**Node Features (per truck)**:
- Position: current (x, y)
- Home depot: depot ID
- Current load: sum of assigned customer volumes (kg)
- Max capacity: truck's volume capacity (kg)
- Load utilization: current_load / max_capacity (%)

**Node Features (per customer)**:
- Location: (x, y)
- volume: delivery volume (kg)
- Visitation status: unvisited or visited

**Graph Edges**:
- **Truck-to-Customer**: Distance from truck's current location to each unvisited customer
- **Truck-to-Depot**: Distance from truck's current location to all depots

---

### ACTION (Fleet-Level Assignment)

**Action Space**: MultiDiscrete array of length T (one action per truck)
- Each element: customer_id ∈ [0, N] or action = N (return to home depot)
- Semantics: Assign one customer to each truck simultaneously
- Example: `action = [cust_3, cust_5, depot, cust_2, ..., depot]`

**Action Constraints**:
- Each customer assigned to at most one truck per step
- Truck can visit customer only if:
  - Truck has remaining capacity: `current_load + customer.volume ≤ truck.max_capacity`
  - Action masking prevents capacity violations (infeasible actions blocked)
- If truck selects action = N (depot): returns to home depot, clears load, resets for next route

---

### REWARD (Minimize Distance + Maximize Utilization)

**Reward Formula**:
```
R(t) = 0.75 * r_routing + 0.25 * r_efficiency
```

**r_routing** (distance cost, applies per assignment):
```
For each truck→customer assignment:
  distance = euclidean_distance(truck.position, customer.location)
  r_routing -= distance
(negative; encourages short routes)
```

**r_efficiency** (volume utilization maximization):
```
For each truck with assigned customers:
  utilization = current_load / truck.max_capacity
  if utilization > 0.8: r_efficiency += 5.0   (strong bonus for full trucks)
  elif utilization > 0.5: r_efficiency += 2.0 (moderate bonus)
  elif utilization < 0.3: r_efficiency -= 1.0 (mild penalty for underfilled)
```

**Terminal Reward** (r_completion):
```
If all customers delivered AND all trucks returned to home depots:
  +500.0 (successful episode)

If customers not delivered at episode end:
  -200.0 per undelivered customer
```

**Total Reward at Step t**:
```
R(t) = 0.75 * r_routing + 0.25 * r_efficiency
(volumes: 75% on distance cost, 25% on utilization)
```

**Action Masking (Feasibility)**:
- Capacity: mask if `current_load + customer.volume > truck.max_capacity`
- Duplicate: mask if customer already assigned this step
- Depot action: always available
- Policy only sees feasible actions → 100% feasibility guarantee

**Episode Termination**:
- **Success**: All N customers delivered + all T trucks at home depots
- **Truncation**: Max steps exceeded (e.g., 1000 steps)

---

## 4. Gymnasium Environment Structure

**Observation Space** (Dict):
```python
{
  "truck_positions": Box(shape=(T, 2), dtype=float32),      # (x, y) per truck
  "truck_loads": Box(shape=(T,), dtype=float32),            # current volume per truck
  "truck_capacities": Box(shape=(T,), dtype=float32),       # max capacity per truck
  "truck_home_depots": Box(shape=(T,), dtype=int32),        # depot ID per truck
  "customer_locations": Box(shape=(N, 2), dtype=float32),   # (x, y) per customer
  "customer_volumes": Box(shape=(N,), dtype=float32),       # volume per customer
  "unvisited_mask": Box(shape=(N,), dtype=int8),            # 1 if unvisited, 0 otherwise
  "feasibility_mask": Box(shape=(T, N+1), dtype=int8)       # 1 if feasible, 0 otherwise
}
```

**Action Space**:
```python
MultiDiscrete([N+1] * T)  # Each truck: customer_id or depot (action=N)
```

---

## 5. Data & Features

- **Trucks**: T trucks with assigned home depots, volume capacities
- **Customers**: N customers with locations and volumes
- **Depots**: D depots with fixed locations
- **Distance Matrix**: Euclidean distance between all locations
- **Objectives**: 
  - Minimize total distance (primary: 75%)
  - Maximize volume utilization (secondary: 25%)

---

## 6. Implementation Steps

### Step 1: Data Generation
```python
# Synthetic instances
customers = [(x, y, volume) for _ in range(N)]
depots = [(x, y) for _ in range(D)]
trucks = [
  {
    "id": t,
    "home_depot": depot_id,
    "max_capacity": capacity
  }
  for t in range(T)
]
```

### Step 2: Environment (`FleetRoutingEnv`)
```python
env = FleetRoutingEnv(
  customers=customers,
  trucks=trucks,
  depots=depots,
  max_steps=1000
)
```

### Step 3: Policy Network
- **Input**: truck states + customer locations/volumes + feasibility masks
- **Architecture**: Transformer encoder + multi-head action selector (one head per truck)
- **Output**: Action probabilities per truck (after masking)

### Step 4: Training
```python
# PPO or A2C with masked action space
for episode in range(num_episodes):
  obs, info = env.reset()
  while not done:
    actions = policy(obs)  # returns masked action distribution
    obs, reward, terminated, truncated, info = env.step(actions)
    # accumulate rewards, train policy
```

### Step 5: Evaluation
- **Metrics**: 
  - Total distance traveled (sum of all truck routes)
  - Average truck utilization (%)
  - % customers delivered
  - Runtime per instance

---

## 7. Training Plan (4 Weeks)

**Week 1: Baseline & Data**
- Implement FleetRoutingEnv
- Generate diverse synthetic instances (N=10→100, T=2→25, D=1→5)
- Establish greedy/nearest-neighbor baseline

**Week 2: First DL Model**
- Implement Transformer + masked action space
- Train with PPO on small instances (N=20, T=5)
- Validate feasibility (100% capacity satisfaction)

**Week 3: Scaling & Optimization**
- Scale to larger instances (N=100, T=20)
- Tune reward volumes (0.75/0.25 ratio)
- Add curriculum learning (size progression)

**Week 4: Evaluation & Polish**
- Benchmark vs. greedy/OR-Tools baselines
- Test generalization to unseen sizes
- Profile runtime and memory usage

---

## 8. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Total Distance** | Sum of distances traveled by all trucks |
| **Avg Utilization** | Mean of (current_load / max_capacity) across trucks |
| **Delivery Rate** | % of customers delivered successfully |
| **Feasibility** | % of episodes with 100% capacity compliance |
| **Speedup** | Inference time vs. greedy baseline |
| **Optimality Gap** | % difference from best-known solution (if available) |

---

## 9. Key Differences from Full Version

| Aspect | Full Version | Simplified |
|--------|------------|-----------|
| **Packing** | 3D packing with pallets | volume-only (no volume) |
| **Time Windows** | Yes | ✅ No |
| **Driver Hours** | 12-hour limit | ✅ No limit |
| **Client Types** | Agrocenter / Other | ✅ All homogeneous |
| **Re-pickup** | Allowed from any depot | ✅ Home depot only |
| **Objectives** | Distance + Time + Vehicles | ✅ Distance + Utilization |
| **Reward volumes** | 0.7/0.2/0.1 | 0.75/0.25 (2 components) |
| **Complexity** | O(T × N × D × features) | ✅ O(T × N × D) |

---

## 10. Code Structure

```
env/
├── types_simple.py             # Customer, Truck, Depot dataclasses
├── utils_simple.py             # Distance, feasibility checks
├── routing_env_simple.py       # SimpleFleetRoutingEnv (Gymnasium)
├── data_generation.py          # Problem instance generation
└── sb3_wrapper.py              # Gymnasium → SB3 wrapper (flattens observations)

model/
├── tsp_agent/
│   └── graph_pointer_network_model.py  # GraphEncoder, GraphPointer, GraphPointerNetwork
├── graph_converter.py          # Convert flat obs → graph representation
├── graph_pointer_policy.py     # GraphAwareFeaturesExtractor + GraphPointerPolicy
└── (other baseline models)

tests/
├── test_graph_pointer_integration.py   # Graph conversion & policy tests
├── test_environment.py                 # Environment fundamentals
├── final_integration_test.py           # End-to-end system tests
├── run_all_tests.py                    # Master test runner
└── README.md                           # Test documentation

train_ppo.py                    # PPO training script
baselines.py                    # Greedy baseline policies
```

---

## 11. Testing

All tests are organized in the `tests/` directory. Run tests with:

### Run all tests (recommended)
```bash
python -m tests.run_all_tests
```

### Run specific test suites
```bash
python -m tests.test_graph_pointer_integration      # Graph Pointer Network tests
python -m tests.test_environment                    # Environment fundamentals
python -m tests.final_integration_test              # End-to-end integration
```

**Expected Results**:
- ✅ Graph Pointer integration: 9/9 tests pass
- ✅ Environment tests: 4/4 test functions pass
- ✅ Final integration: 5/6 tests pass (PPO model loading skipped if no trained model exists)

For detailed test documentation, see `tests/README.md`

---

## 12. Extensions (Future)

- **V2**: Add time windows (customer availability)
- **V3**: Add driver hour constraints
- **V4**: Add 3D packing with pallets
- **V5**: Add multi-product types with client preferences
- **V6**: Dynamic routing (orders arrive during day)

---

## 13. Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Low utilization bonus | volumes too imbalanced | Increase r_efficiency coefficient |
| Slow convergence | Sparse rewards | Add intermediate rewards for partial deliveries |
| Generalization failure | Overfitting to one size | Use curriculum learning (N: 10→20→50→100) |
| Capacity violations | Masking not working | Verify feasibility_mask computation before softmax |

---

## 14. Quick Start Example

```python
from env.routing_env_simplified import FleetRoutingEnv
from env.config import create_simple_scenario
import gymnasium as gym

# Create scenario
customers, trucks, depots = create_simple_scenario(
  num_customers=20,
  num_trucks=5,
  num_depots=1
)

# Create environment
env = FleetRoutingEnv(
  customers=customers,
  trucks=trucks,
  depots=depots,
  max_steps=500
)

# Random baseline
obs, info = env.reset()
for step in range(100):
  # Sample random feasible actions
  action = env.action_space.sample()  # respects masking automatically
  obs, reward, terminated, truncated, info = env.step(action)
  
  if terminated or truncated:
    print(f"Episode done. Total distance: {info['total_distance']:.2f}")
    print(f"Avg utilization: {info['avg_utilization']:.2%}")
    obs, info = env.reset()

env.close()
```

---

**Version History**:
- v1.0 (2026-01-17): Initial simplified version - volume-only, no time constraints, home-depot-only



┌─────────────────────────────────────────────────────────────┐
│                    AGENT (Decision Maker)                   │
│                                                             │
│  ┌──────────────────────┐         ┌──────────────────────┐ │
│  │   POLICY             │         │  NEURAL NETWORK      │ │
│  │  (Decision Logic)    │◄────────│  (MlpPolicy)         │ │
│  └──────────┬───────────┘         └──────────────────────┘ │
│             │                                              │
│      Takes observations,                                   │
│      outputs actions                                       │
└─────────────┼──────────────────────────────────────────────┘
              │
              │ action
              ▼
┌─────────────────────────────────────────────────────────────┐
│              ENVIRONMENT (Fleet Routing World)              │
│                                                             │
│  ├─ Customers & Trucks                                    │
│  ├─ Feasibility masking                                   │
│  ├─ Reward calculation                                    │
│  └─ Observation generation                                │
└─────────────┬──────────────────────────────────────────────┘
              │
              │ observation, reward, done
              ▼
         (back to Agent)


The Loop:
  ENVIRONMENT → sends OBSERVATION to agent
  AGENT uses POLICY to decide on ACTION
  AGENT sends ACTION to environment
  ENVIRONMENT executes action, computes REWARD, produces new OBSERVATION
  POLICY learns from (observation, action, reward) experiences
  Repeat until done



## 15. Suggested Workflow

### 1. Run all tests
```bash
python -m tests.run_all_tests
```

### 2. Train a policy
```bash
python train_ppo.py --timesteps 5000 --customers 15 --trucks 4
```

### 3. Evaluate performance
```bash
python train_ppo.py --timesteps 100000 --customers 20 --trucks 6 --lr 3e-4
```

For more training options, see `train_ppo.py --help`