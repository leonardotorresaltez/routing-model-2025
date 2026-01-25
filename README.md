
# Final Project: Deep Learning for Logistics Optimization
**Title:** Maximize Truck volume Utilization with Deep Learning
**Date:** 2026-01-19
**Version:** 6  

---

**Main objectives:**
1. Deliver goods from depots to all customers, with no deadline constraints.
2. Minimize total travel distance for the fleet.
3. Maximize volume utilization on each truck.

**V0
1. For 1 truck, deliver all the goods with the less distance as possible

## 1. Problem Statement
Design a system that:
- Optimizes routing for all vehicles as a single fleet 
- Starts the day at a depot.
- Delivers goods from depots to their destinations (customers).
- Packs goods properly in the pallets (3D packing constraints).
- Packs pallets properly in the truck (3D packing constraints).
- May go to pick up goods at a new depot nearby the last delivery of the last route.
- Ends the day (deliveries) at the starting point depot.

**V0
- Optimizes routing for a single vehicle instead of the entire fleet. 
- Starts the day at a depot.
- Delivers goods to all destinations/customers.


**Key Insight**: We make parallel routing decisions across all trucks in a single episode, minimizing **global fleet cost** (total distance + time) while maximizing truck utilization (weight + capacity loading). Vehicle count is not explicitly minimized; instead, it emerges naturally as the cost function incentivizes fuller trucks.


---

# Logistics Requirements Document

### 1. Resources
- **Depots**:
  - D number of depots (D loading points).
- **Trucks**:
  - Each depot has its own fleet of trucks, with different types of trucks
  - Different truck volumes and different maximum truck weights.
  - Trucks work **less than 12 hours per day**.
  - Trucks can have more than 1 route, with each route having multiple deliveries per day.
  - Not all trucks can access all destinations (due to farm access restrictions, see *Client* paragraph).
  - **Types of trucks**:
    i.e. Type A: Max capacity and volume.

**V0
- **Depots**:
  - D number of depots (D loading points). D =1.
- **Trucks**:
  - 1 truck that can deliver as many goods as it is asked for. 

---

### 2. Pallets (V2)
- Different pallet volumes and different maximum pallet weights.
  - Pallet of type A: base weight and volume.
  - Pallet of type B: base weight and volume.
- Each pallet type, in case of having unique type of goods, has a predefined maximum number of goods.
  - Example: Pallet A, goods = Potatoes, max number = 20.
- For pallets with mixed good types, **maximum 35 units/sacks per pallet** (for simplification).

**V0
- No pallets
---

### 3. Products
- List of product weights and volumes:
  - Example: Product 1 → weight = 3 kg, volume = 10×2×30 cm.

**V0
- Not rellevant, since the truck can take as many goods as possible, from any weight and size.
---

### 4. Delivery Constraints (V2)
- **Goods of type A** (agrocenter client, milk product):
  - Must be served within **2 days** from the order day.
- All other goods:
  - Must be served within **7 days** from the order day.

**V0
- Not rellevant.
---
 
### 5. Delivery/Clients (V2)
- **Access types**:
  - All trucks.
  - Trucks type A and B.
  - Only trucks type B.
- **Client types**:
  - Agrocenter.
  - Others.

**V0
- Not rellevant.
---

## 2. Why This Problem?
- Real-world relevance: combines **Vehicle Routing Problem (VRP)** with **3D Bin Packing**.
- Multi-objective optimization: fleet-wide routing + truck utilization maximization.
- **Key difference from classical routing**: We optimize the entire fleet cost simultaneously, allowing the model to learn load-balancing across trucks dynamically.



---

## 3. Project Tracks
### Reinforcement Learning (Train from Scratch)

- **Environment**:
  - **STATE** (Multi-Agent, Multi-Depot Graph)
    Represent the problem as a dynamic graph with all T trucks and all undelivered customers:

    *Graph Nodes*:
    - **Truck Nodes** (T total, one per truck):
      - Current location: (x, y) coordinate or depot/depot node
      - Home depot: which of D depots the truck belongs to (return point at day end)
    
    - **Depot Nodes**  (D static):
      - Fixed loading point locations (x, y) 
      - Truck bases (each truck assigned to one depot)
      - Optional re-pickup points (trucks may visit nearby depots after deliveries)
    
    - **Customer Nodes** (N undelivered customers):
      - Location: (x, y) coordinate
      - Demand: quantity (units) + weight (kg) + volume (m³)
      - Time window: [ready_time, due_time] (minutes from day start)
      - Truck access restrictions: [can_A, can_B, can_C]
      - Client type: agrocenter vs. other (affects SLA urgency)

    - **Already visited nodes**  
      - Locations: (x, y)        
  
      **V0:
      - Truck curent location: (x, y) coordinate
      - already visited nodes
      - Depot and Customer nodes (x,y)



    *Node Features (per truck)*:
    - Position: current (x, y) or depot ID
    - Truck type: A, B, or C (determines capacity and access permissions)
    - Home depot: depot ID (0–D) where truck returns at end of day
    - Capacity state:
      - Current load: sum of all assigned customer demands (units)
      - Current weight: sum of assigned customer weights (kg)
      - Current volume: sum of assigned customer volumes (m³)
    - Time state:
      - Elapsed time: total time spent (travel + service + wait, in minutes)
      - Max hours: 12 hours (720 minutes) hard limit
    - Route state: list of visited customer IDs (for this truck's current route)
  
    **V0:
    - Position: current (x, y) 



    *Node Features (per customer)*:
    - Geographic location: (x, y) in coordinate space
    - Demand vector: [quantity, weight_kg, volume_m3]
    - Time window: [ready_time_min, due_time_min] relative to day start
    - Truck compatibility: which truck types can serve (access restrictions)
    - Client type: binary flag (agrocenter=1, other=0)
    - Visitation status: unvisited, visited by truck_i, or completed
  
    **V0:
    - Position: current (x, y) 
    

    *Graph Edges*:
    - **Truck-to-Customer edges**: 
      - Distance/time from truck's current location to each unvisited customer
      - Feasibility mask: 1 if feasible (truck type access + capacity + time window), 0 otherwise
    - **Truck-to-depot edges**:
      - Distance/time from truck's current location to any of the D depots (for return or re-pickup)
      - Feasibility: always available (to end route or re-pickup)
    - **Customer-to-depot edges** (for re-pickup feasibility):
      - Distance/time from customer location to nearby depots
      - Used to determine if truck can feasibly reach a depot for re-pickup after delivery

    **V0:
    - **node-to-node edges**: 
      - Distance/time from truck's current location to next one

  - **ACTION** (Fleet-Level Customer Assignment)
    Action space: **MultiDiscrete array of length**
    - Each element: customer_id ∈ [0, N] or action = N (return to depot)
    - Semantics: assign one customer (or depot action) to each truck **simultaneously** in a single step
    - Example: 
      - Truck 0 → visit customer 3
      - Truck 1 → visit customer 7
      - Truck 1 → return to its home depot
      - Truck 2 → visit customer 2
      - ... (T trucks total)
      - Truck T-1 → return to its home depot

    *Constraints on Actions*:
    - Each customer assigned to at most one truck per step (no duplicate assignments)
    - Truck can only visit a customer if feasible:
      - Truck type matches customer access requirements
      - Truck has remaining capacity (weight, volume, load count)
      - Truck has remaining time (won't exceed 12-hour limit)
      - Customer's time window is still open (arrival_time ≤ due_time)
    - If truck selects action = N (depot), it returns to its home depot, clears its load, and resets for next route
    - All feasibility checks are **masked at the policy level** (infeasible actions blocked before softmax)

    -*Packing Decision*:
    - Given selected customer's goods, decide which pallet type to use
    - Decide item placement within pallet (heuristic guillotine or learned policy)
    - Decide which truck route the goods go into (if multi-route)
  
    **V0:
    Action space: 
    - Each element: customer_id ∈ [1] or action = N (return to depot)
    - Example: 
      - Truck 0 → visit customer 3
      - Truck 0 → visit customer 5
      - Truck 0 → return to its home depot

    *Constraints on Actions*:
    - Each customer assigned to at most one truck per step (no duplicate assignments)
    - Truck can only visit a customer if feasible:
    - If truck selects action = N (depot), it returns to its home depot, clears its load, and resets for next route
    - All feasibility checks are **masked at the policy level** (infeasible actions blocked before softmax)


  - **REWARD** (Global Fleet Cost Minimization)
    Multi-component scalar reward (normalized weighted sum):

    **R(t) = A · r_routing + B · r_efficiency + C · r_completion** 
    
    A/B/C are reward weights, these are empirical hyperparameters (to be tuned and validated during training)

    **V0:
    **R(t) = A · r_routing** 


    **Key Principle**: All hard constraints (capacity, time window, truck access) are enforced **exclusively via action masking**. The reward function focuses only on optimizing cost and utilization. This prevents the policy from learning infeasible assignments in the first place.

    **r_routing** (travel cost penalty, applies per assignment):
    ```
    For each truck→customer assignment in this step:
      distance = euclidean_distance(truck.position, customer.location)
      travel_time = distance / avg_speed (minutes)
      r_routing -= 0.1 * travel_time  # where travel_time = distance / avg_speed

    (negative cost; encourages short, fast deliveries)
    ```

    **r_efficiency** (truck utilization maximization):
    Encourage loading trucks to high capacity. Vehicle count is NOT explicitly minimized; it emerges naturally from cost optimization.
    ```
    For each truck with assigned customers:
      utilization = current_load / truck.max_capacity
      if utilization > 0.8: r_efficiency += 5.0   (strong bonus for well-loaded trucks)
      elif utilization > 0.5: r_efficiency += 2.0 (moderate bonus for adequately loaded trucks)
      elif utilization < 0.3: r_efficiency -= 1.0 (mild penalty for significantly underfilled trucks)
    ```
    **Rationale**: Maximize how full each truck is, not how many trucks are used. The distance/time cost in r_routing naturally incentivizes fuller trucks: adding an underutilized truck increases total distance/time cost, so the policy learns to consolidate loads.

    **r_completion** (terminal reward, only at episode end):
    - If all customers delivered AND all trucks returned to home depots:
      ```
      r_completion = +500.0 (successful episode)
      ```
    - Bonus for early/on-time delivery:
      ```
      For each customer delivered before due_time:
        r_completion += 20.0
      ```
    - Penalty for missed deliveries:
      ```
      For each customer not delivered at episode end:
        r_completion -= 200.0
      ```

    **Action Masking (Feasibility Enforcement)**:
    Before the policy outputs action probabilities, all infeasible (truck, customer) pairs are masked:
    - ✅ Truck type access: mask if customer not in truck's allowed types [A, B, C]
    - ✅ Capacity constraint: mask if assigning customer would exceed truck's max weight/volume/load
    - ✅ Time window constraint: mask if truck cannot arrive before customer's due_time
    - ✅ Duplicate prevention: mask if customer already assigned this step to another truck
    - The policy only sees feasible actions, so it learns **only valid assignments**
    - The softmax and sampling are applied **after masking**, ensuring 100% feasibility
   
   

    **Total Reward at Step t**:
    ```
    R(t) = A * r_routing + B * r_efficiency + C * r_completion
    (weights normalized; adjust empirically during training)
    All components assume masked, feasible actions only.
    ```
    **V0: 
    R(t) = A * r_routing





    **Episode Termination Conditions**:
    - **Success**: All N customers delivered + all T trucks returned to their home depots
    - **Truncation**: Max steps exceeded (e.g., 2000 steps for large instances) → incomplete episode
    - **Feasibility Guarantee**: With action masking, all delivered customers satisfy time windows, capacity, and access constraints
    
    **V0:
    - **Success**: All N customers delivered by 1 truck T
    - **Truncation**: Max steps exceeded (e.g., 50 steps for large instances) → incomplete episode





- **Policy Architecture**:
  - Transformer/Pointer Network for constructive routing.
  - GNN for relational constraints and edge scoring.
- **RL Algorithm**:
  - PPO or Actor–Critic (policy gradient with baseline).
- **Reward**:
  - Negative travel time/distance.
  - Penalty for infeasible packing or constraint violations.
  - Bonus for completing all jobs with zero violations.
- **Note**: 
  - Modular: Routing RL can be trained independently; packing added later
  - GNN-friendly: Graph structure naturally encodes customer relationships
  - Feasibility-first: Hard constraints (masking + large penalties) ensure no violations escape to production
  - Real-world aligned: Reflects actual dispatch decisions (pick next customer) + cost objectives




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
- Objectives: minimize cost (distance + time), maximize truck utilization, respect SLA lateness constraints, CO₂/fairness weights (V3+).

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
- Combine routing (and V2-packing feasibility checks).
- Run validator/repair; then a short OR local search for final quality.

### Step 5: Evaluation
- Metrics: total cost (distance + time), average truck utilization, % feasible routes, lateness.
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
- **Routing**: total distance/time cost, average truck utilization (%), lateness, missed windows (must be zero after repair).
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
- Packing goods into pallets.
- Time windows and driver hours-of-service.
- Learned packing policy and dock scheduling.

---

## 12. Common Pitfalls & Mitigations
- **Feasibility leakage** → Always run validator + repair; never ship routes with violations.
- **Distribution shift** → Train on mixed synthetic + real; schedule monthly fine-tunes.
- **Overfitting to size** → Curriculum across instance sizes; test generalization.
``


---
## 13. Simplifications
  Only 1 truck, and 1 depot
  Remove packing/pallets section
  Remove client type distinctions
  Remove time windows
  Remove driver hour limits
  Simplify the truck state (no elapsed time, no time window constraints)
  Simplify customer features (no time windows, no client type)
  Simplify action masking (no time window checks, no hour limit checks)
  Simplify reward functions (remove time-based penalties)
