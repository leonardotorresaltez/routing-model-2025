import argparse
from dataclasses import dataclass

@dataclass
class Config:
    # --- Experiment ---
    project_name: str = "routing-model-2025"
    run_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    wandb: bool = True  # Toggle W&B logging
    data_dir: str = "data_version_1"  # data path
    debug = False
    
    # --- Model ---
    embed_dim: int = 128
    max_daily_delivery_time_each_truck: int = 24  # hours
    
    # --- Training ---
    lr: float = 1e-3
    episodes: int = 300 if not debug else 2
    log_interval: int = 20
    gamma: float = 0.95  # Discount factor for rewards
    entropy_bonus: float = 0.01  # Coefficient for entropy bonus to encourage exploration
    
    def __post_init__(self):
        if self.data_dir == "data_version_2":
            self.max_daily_delivery_time_each_truck = 24
            
        else:
            self.max_daily_delivery_time_each_truck = 12
        
        # Construct project_name
        self.project_name = f"{self.project_name}_{self.data_dir}"
        # Construct Run Name
        self.run_name = f"lr{self.lr}_gamma{self.gamma}_sd{self.seed}"

def parse_args() -> Config:
    base_cfg = Config()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, default=base_cfg.lr)
    parser.add_argument("--episodes", type=int, default=base_cfg.episodes)
    parser.add_argument("--embed_dim", type=int, default=base_cfg.embed_dim)
    parser.add_argument("--seed", type=int, default=base_cfg.seed)
    parser.add_argument("--device", type=str, default=base_cfg.device)
    parser.add_argument("--data_dir", type=str, default=base_cfg.data_dir)
    parser.add_argument("--max_daily_delivery_time_for_each_truck", type=int, default=base_cfg.max_daily_delivery_time_each_truck)
    
    # Flag: --no-wandb to disable logging
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    
    return Config(
        lr=args.lr,
        episodes=args.episodes,
        embed_dim=args.embed_dim,
        seed=args.seed,
        device=args.device,
        wandb=not args.no_wandb,
        data_dir=args.data_dir
    )