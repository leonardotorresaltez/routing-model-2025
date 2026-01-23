from dataclasses import dataclass
import argparse

@dataclass
class Config:
    # Experiment
    project_name: str = "logistics-rl-poc"
    run_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    wandb: bool = True

    # Environment
    num_nodes: int = 10
    num_trucks: int = 3
    depot: int = 0
    max_steps: int = 100
    graph_mode: bool = True

    # Model
    embed_dim: int = 128

    # Training
    lr: float = 1e-3
    episodes: int = 500
    log_interval: int = 20


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--trucks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    run_name = f"N{args.nodes}_T{args.trucks}_lr{args.lr}_sd{args.seed}"

    return Config(
        num_nodes=args.nodes,
        num_trucks=args.trucks,
        lr=args.lr,
        episodes=args.episodes,
        embed_dim=args.embed_dim,
        seed=args.seed,
        device=args.device,
        wandb=not args.no_wandb,
        run_name=run_name
    )