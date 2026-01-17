# Test Suite for Fleet Routing with Graph Pointer Network

This directory contains all tests for the fleet routing reinforcement learning system.

## Test Files

### 1. `test_graph_pointer_integration.py` (9 tests)
Tests the Graph Pointer Network integration with SB3 PPO.

**What it tests:**
- Graph converter functions (observation reconstruction, node features, adjacency matrix)
- GraphPointerNetwork execution
- GraphAwareFeaturesExtractor feature extraction pipeline
- GraphPointerPolicy instantiation
- Integration with actual routing environment
- Policy forward pass with real observations

**Run individually:**
```bash
python -m tests.test_graph_pointer_integration
```

---

### 2. `test_environment.py` (4 test functions)
Tests the simplified fleet routing environment fundamentals.

**What it tests:**
- Environment initialization and reset
- Observation structure validation
- Feasibility masking enforcement (capacity constraints)
- Full episode completion
- Baseline policy execution (Greedy Nearest, Nearest Depot First, Furthest Customer)

**Run individually:**
```bash
python -m tests.test_environment
```

---

### 3. `final_integration_test.py` (6 test stages)
End-to-end integration test of the complete system.

**What it tests:**
- Data generation
- Environment creation
- SB3 wrapper functionality
- Baseline policy execution
- PPO model loading and inference (if model exists)
- Feasibility masking validation

**Run individually:**
```bash
python -m tests.final_integration_test
```

---

### 4. `run_all_tests.py` (Master runner)
Orchestrates all tests in sequence.

**Run complete test suite:**
```bash
python -m tests.run_all_tests
```

---

## Quick Start

### Run all tests (recommended)
```bash
python -m tests.run_all_tests
```

### Run specific test
```bash
python -m tests.test_graph_pointer_integration
python -m tests.test_environment
python -m tests.final_integration_test
```

### Run with verbose output
```bash
python -m tests.run_all_tests
```

---

## Test Results

All tests should pass with output similar to:

```
============================================================
Testing Graph Converter Functions
============================================================
[PASS] reconstruct_observation_components
[PASS] build_node_features
[PASS] build_adjacency_matrix
[PASS] observation_to_graph
[PASS] create_networkx_graph
```

### Expected Outcomes

| Test | Expected Result | Purpose |
|------|-----------------|---------|
| Graph Pointer Integration | PASS | Validates graph conversion and feature extraction |
| Environment Basics | PASS | Confirms environment initializes correctly |
| Feasibility Masking | PASS | Ensures capacity constraints are enforced |
| Episode Completion | PASS | Verifies episodes terminate properly |
| Baseline Policies | PASS | Establishes performance benchmarks |
| Data Generation | PASS | Validates problem instance creation |
| SB3 Wrapper | PASS | Confirms observation flattening works |
| Baseline Execution | PASS | Tests heuristic policies |
| PPO Inference | SKIP (if model missing) | Validates trained policy loading |
| Feasibility (Integration) | PASS | Final validation of constraints |

---

## Troubleshooting

### Graph Pointer tests fail
- Ensure torch, numpy, and networkx are installed
- Verify `model/graph_converter.py` exists
- Check that `model/tsp_agent/graph_pointer_network_model.py` is accessible

### Environment tests fail
- Ensure `env/routing_env_simple.py` exists
- Verify `env/types_simple.py` is accessible
- Check that `baselines` module is available

### Integration tests fail
- Ensure all environment components are present
- Verify `env/sb3_wrapper.py` is accessible
- Check stable-baselines3 installation

### PPO model test skipped
- This is normal if `models/ppo_routing_final.zip` doesn't exist
- Train a model first: `python train_ppo.py`

---

## Test Metrics

### Baseline Performance (typical for 15 customers, 4 trucks)
- **Greedy Nearest Policy**: 800-1000 distance units
- **Optimal bounds**: ~800-1600 distance units
- **Average utilization**: 55-65%
- **Episode length**: 8-15 steps

### Graph Pointer Feature Extraction
- **Feature dimension**: 64
- **Graph nodes**: T + N (trucks + customers)
- **K-neighbors**: 5 (by default)
- **Processing time per batch**: ~50-100ms (CPU)

---

## Architecture Overview

```
Tests
├── test_graph_pointer_integration.py
│   ├── Graph conversion tests
│   ├── Feature extraction tests
│   └── Policy instantiation tests
│
├── test_environment.py
│   ├── Environment basics
│   ├── Feasibility masking
│   ├── Episode completion
│   └── Baseline policies
│
├── final_integration_test.py
│   ├── Data generation
│   ├── Environment creation
│   ├── Wrapper functionality
│   ├── Baseline execution
│   ├── Model inference
│   └── Constraint validation
│
└── run_all_tests.py (orchestrator)
    └── Runs all above sequentially
```

---

## Next Steps

After tests pass:

1. **Train a new policy**:
   ```bash
   python train_ppo.py --timesteps 100000 --customers 15 --trucks 4
   ```

2. **Evaluate trained policy**:
   ```bash
   python train_ppo.py --timesteps 0  # Loads and evaluates existing model
   ```

3. **Experiment with parameters**:
   ```bash
   python train_ppo.py --timesteps 50000 --customers 20 --trucks 6 --lr 5e-4
   ```

---

## Contributing

When adding new tests:
1. Place test functions in appropriate file
2. Follow naming convention: `test_<feature_name>()`
3. Add docstring explaining what is tested
4. Update this README with new test information
5. Verify all tests pass before committing
