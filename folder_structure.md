# Logistics RL: Routing Optimization POC

This repository contains a modular Reinforcement Learning framework designed to solve Routing Problems (like TSP and VRP) using an Attention-based Pointer Network.

## Project Structure
```
logistics_rl/
├── checkpoints/             # Saved model weights (.pt files)
├── configs/
│   ├── __init__.py
│   └── config.py            # Hyperparameters & CLI parsing
├── core/
│   ├── envs/
│   │   ├── __init__.py
│   │   └── tsp_env.py       # Gymnasium Environment logic
│   └── models/
│       ├── __init__.py
│       ├── agent.py         # Weights update logic
│       └── policy.py        # Attention/Pointer Network architecture
├── data/                    
├── main.py                  # Entry point for single training runs
├── run_experiments.py       # Grid search orchestrator
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```
---

## Installation & Setup

1. **Install Dependencies**
   Use the provided requirements file to set up your environment:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Weights & Biases (Optional)**
   This project is integrated with W&B for real-time experiment tracking.
   ```bash
   wandb login
   ```

---

## How to Run

### 1. Single Experiment
To run a specific configuration, use `main.py`. You can override defaults using flags:
```bash
python main.py --nodes 20 --lr 0.001 --episodes 1000
```

### 2. Run a Grid Search (Loop)
To test multiple combinations of parameters automatically, use the orchestrator:
```bash
python run_experiments.py
```

### 3. Disable Tracking
If you want to run a quick test without logging to W&B:
```bash
python main.py --no-wandb
```

---

## Model Architecture
The policy uses a **Pointer Network** approach. Instead of traditional attention that "mixes" values, this model uses the attention scores directly as a probability distribution over the available nodes.

1. **Encoder:** Embeds (x, y) coordinates into a d-dimensional space.
2. **Query:** The embedding of the current location.
3. **Keys:** The embeddings of all possible destination nodes.
4. **Masking:** A boolean mask is applied to ensure the agent never visits a node twice.

---

## Monitoring Results
All metrics (Reward, Loss, Episode Length) are sent to **Weights & Biases**. You can compare different runs (e.g., Learning Rate 0.001 vs 0.0001) directly in your web browser.