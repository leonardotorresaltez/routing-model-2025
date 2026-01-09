
# Final Project: Deep Learning for Logistics Optimization
**Title:** Deliver Goods Efficiently with Deep Learning  
**Date:** 2026-01-09  
**Version:** 3  

---

**Main objectives:**
1. Deliver the goods to the clients before the deadline.
2. Minimize total travel distance/time (costs) for the company.

## 1. Problem Statement
Design a system that:
- Starts the day at a depot.
- Delivers goods to their destinations.
- Packs goods properly in the pallets (3D packing constraints).
- Packs pallets properly in the truck (3D packing constraints).
- May go to pick up goods at a new depot nearby the last delivery of the last route.
- Ends the day (deliveries) at the starting point depot.

---

# Logistics Requirements Document

### 1. Resources
- **Factories**:
  - 5 factories (5 loading points).
- **Trucks**:
  - Each factory has 5 trucks: 
    - factory1: 1 of type A, 3 of type B, 1 of type C.
    - factory2-5: 2 of type A, 2 of type B, 1 of type C.
  - Different truck volumes and different maximum truck weights.
  - Trucks work **less than 12 hours per day**.
  - Trucks can have more than 1 route, with each route having multiple deliveries per day.
  - Not all trucks can access all destinations (due to farm access restrictions, see *Client* paragraph).
  - **Types of trucks**:
    - Type A: Max capacity and volume → *Pending Viviana*.
    - Type B: Max capacity and volume → *Pending Viviana*.
    - Type C: Max capacity and volume → *Pending Viviana*.

---

### 2. Pallets
- Different pallet volumes and different maximum pallet weights.
  - Pallet of type A: base weight and volume → *Pending Viviana*.
  - Pallet of type B: base weight and volume → *Pending Viviana*.
- Each pallet type, in case of having unique type of goods, has a predefined maximum number of goods.
  - Example: Pallet A, goods = Potatoes, max number = 20 → *Pending Viviana*.
- For pallets with mixed good types, **maximum 35 units/sacks per pallet** (for simplification).

---

### 3. Products
- List of product weights and volumes:
  - Example: Product 1 → weight = 3 kg, volume = 10×2×30 cm → *Pending Viviana*.
- If list is not available, approximate total number of products, minimum and maximum volume, and minimum and maximum weight.

---

### 4. Delivery Constraints
- **Goods of type A** (agrocenter client, milk product):
  - Must be served within **2 days** from the order day.
- All other goods:
  - Must be served within **7 days** from the order day.

---

### 5. Delivery/Clients
- **Access types**:
  - All trucks.
  - Trucks type A and B.
  - Only trucks type B.
- **Client types**:
  - Agrocenter.
  - Others.

---

## 2. Why This Problem?
- Real-world relevance: combines **Vehicle Routing Problem (VRP)** with **3D Bin Packing**.
- Multi-objective optimization: routing + packing + capacity + SLA.
- Great for showcasing deep learning techniques beyond classical solvers.


---

## 3. Project Tracks
### Track A — Reinforcement Learning (Train from Scratch)
- **Environment**:
  - State: current location, remaining jobs, truck load state, remaining capacity/time.
  - Action: choose next stop (delivery); optional packing placement.
- **Policy Architecture**:
  - Transformer/Pointer Network for constructive routing.
  - GNN for relational constraints and edge scoring.
- **RL Algorithm**:
  - PPO or Actor–Critic (policy gradient with baseline).
- **Reward**:
  - Negative travel time/distance.
  - Penalty for infeasible packing or constraint violations.
  - Bonus for completing all jobs with zero violations.
- **Note**: Always wrap with a **validator + repair** step for hard constraints.

### Track B — Fine-Tuning a Neural Heuristic
- Start from a pre-trained sequence model (Transformer).
- Fine-tune on synthetic and solver-generated routes (knowledge distillation).
- Penalize packing/timing infeasibility during fine-tune.
- Outcome: faster convergence; near-solver quality at inference-time speed.

### Track C — Retrieval-Augmented Generation (RAG)
- Use a knowledge base of routing rules and packing heuristics.
- Retrieve relevant examples/constraints.
- Generate explanations and **suggest repairs** using an LLM.
- Outcome: transparency and operator-in-the-loop workflows.

> **New Track D — GNN-Guided Heuristics**
> Learn to score edges/customers/moves; plug scores into a classical local search (ruin & recreate, 2-opt/3-opt, relocate, swap). Gains speed without sacrificing feasibility.

---

## 4. Recommended Approach (Production-Proven)
**Hybrid: DL Constructor → Feasibility Repair → OR Refinement**
1. **Neural Constructor**: Transformer encoder–decoder with constraint-aware masking for capacity, time windows, and delivery precedence.
2. **Training**: Policy gradient (REINFORCE with self-critic) + curriculum on instance sizes and constraint densities.
3. **Validator/Repair**: Enforce hard constraints; insert/relocate to fix time-window or precedence issues; packing feasibility check.
4. **OR Local Search (short budget)**: Guided Local Search; neighborhood moves (2-opt, 3-opt, relocate, swap) for 1–10 seconds.
5. **Distillation (optional)**: Fine-tune the constructor on OR-refined routes to close the optimality gap while retaining speed.

**When to use what**
- Static VRP with tight constraints → **OR-only** baseline first.
- Many similar instances or latency-critical decisions → **DL + Repair + OR** hybrid.
- Dynamic/online routing → **RL dispatch/insertion + rolling-horizon OR**.
- Very large graphs → **GNN-guided neighborhoods** to scale.

---

## 5. Data & Features
- Job data: demand, delivery locations, ready/due day, service times.
- Vehicles: capacities, skills, compatibility, breaks/hours.
- Distance/Time: matrices by time-of-day or features from a road graph.
- Packing: item dimensions, weight, stacking rules, fragility, orientation.
- Objectives: cost, lateness penalties, #vehicles, CO₂/fairness weights.

---

## 6. Steps & Algorithms
### Step 1: Data Generation
- Synthetic instances: random coordinates, deliveries, item dimensions.
- Curriculum: sizes from N=50…500; vary constraint densities.

### Step 2: Packing Module
- Heuristic: shelf or guillotine algorithms for 3D packing.
- Optional: small neural policy for item placement; integrate feasibility checks.

### Step 3: Routing Module
- DL constructor (Transformer/PointerNet) or RL policy.
- Constraint masking for capacity, time windows, and precedence.

### Step 4: Integration
- Combine routing and packing feasibility checks.
- Run validator/repair; then a short OR local search for final quality.

### Step 5: Evaluation
- Metrics: total cost, % feasible routes, packing utilization, vehicles used, lateness.
- Runtime distribution (p50, p95); per-instance gap to best-known baseline.

---

## 7. Training & Benchmark Plan (6–8 Weeks)
**Weeks 1–2: Problem framing & baselines**
- Freeze constraints/objective; build a simulator/generator aligned with real distributions.
- Establish OR baselines (e.g., CP-SAT/Guided Local Search) and log metrics.

**Weeks 3–4: First DL model**
- Implement attention model with constraint masks (capacity, time windows, precedence).
- Train with REINFORCE + self-critic on mixed synthetic/real instances.
- Add feasibility repair and 1–5s local search.

**Weeks 5–6: Hybrid improvements**
- Distill OR-refined routes (imitation learning).
- Add features: time-of-day travel times, service priorities, soft penalties.
- Target <3–5% optimality gap with 10×–100× speedup vs OR-only.

**Weeks 7–8: Production readiness**
- Guardrails: hard-constraint validator, uncertainty-based fallback to OR-only.
- Monitoring: drift detection and monthly fine-tuning with operational data.

---

## 8. Evaluation Metrics
- **Routing**: total distance/time, vehicles used, lateness, missed windows (must be zero after repair).
- **Packing**: feasibility rate, utilization, violation types (overstack, orientation).
- **Operational**: runtime p50/p95, % instances solved within SLA, gap to baseline.
- **Robustness**: stress tests—peak loads, extreme time windows, long-tail distances; generalization to new depots/sizes.

---

## 9. Code Skeletons
*(Production-friendly with type hints, dataclasses, and logging)*  
Includes:
- Constraint-aware policy (PyTorch-style).
- Training loop (policy gradient skeleton).
- Packing & integration skeleton.
- OR refinement conceptual notes.

---

## 10. Deliverables
- Code: environment, policy, packing module, validator, OR integration hooks.
- Report: algorithm choices, training curves, ablations, evaluation vs baselines.
- Demo: route visualization and packing layout.

---

## 11. Extensions
- Multi-truck, multi-depot, heterogeneous fleet.
- Time windows and driver hours-of-service.
- Learned packing policy and dock scheduling.

---

## 12. Common Pitfalls & Mitigations
- **Feasibility leakage** → Always run validator + repair; never ship routes with violations.
- **Distribution shift** → Train on mixed synthetic + real; schedule monthly fine-tunes.
- **Overfitting to size** → Curriculum across instance sizes; test generalization.
- **Opaque objectives** → Document scalarization and trade-offs; include fairness/CO₂ as explicit terms.
- **Latency spikes** → Batch inference; maintain OR-only fallback for edge cases.
``
