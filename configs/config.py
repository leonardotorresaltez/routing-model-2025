import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # --- Experiment ---
    project_name: str = "logistics-rl-poc"
    run_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    wandb: bool = True  # Toggle W&B logging
    data_dir: str = "data_version_2"  # data path
    
    # --- Model ---
    embed_dim: int = 128
    
    # --- Training ---
    lr: float = 1e-3
    episodes: int = 50 # 500
    log_interval: int = 20

def parse_args() -> Config:
    
    base_cfg = Config()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, default=base_cfg.lr)
    parser.add_argument("--episodes", type=int, default=base_cfg.episodes)
    parser.add_argument("--embed_dim", type=int, default=base_cfg.embed_dim)
    parser.add_argument("--seed", type=int, default=base_cfg.seed)
    parser.add_argument("--device", type=str, default=base_cfg.device)
    parser.add_argument("--data_dir", type=str, default=base_cfg.data_dir)

    
    # Flag: --no-wandb to disable logging
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()
    
    # Construct Run Name
    run_name = f"{args.data_dir}_lr{args.lr}_sd{args.seed}"
    
    return Config(
        lr=args.lr,
        episodes=args.episodes,
        embed_dim=args.embed_dim,
        seed=args.seed,
        device=args.device,
        wandb=not args.no_wandb,
        run_name=run_name,
        data_dir=args.data_dir,
    )