import argparse
from dataclasses import dataclass

@dataclass
class Config:
    # --- Experiment ---
    project_name: str = "routing-model-2025"
    run_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    wandb: bool = True
    data_dir: str = "data_version_1" 
    
    embed_dim: int = 128
    max_daily_delivery_time_each_truck: int = 24

    lr: float = 5e-5
    episodes: int = 500
    log_interval: int = 20
    
    def __post_init__(self):
        # Now self.data_dir exists and can be checked
        if self.data_dir == "data_version_1":
            self.max_daily_delivery_time_each_truck = 24
        else:
            self.max_daily_delivery_time_each_truck = 12
        
        self.project_name = f"{self.project_name}_{self.data_dir}"
        self.run_name = f"lr{self.lr}_sd{self.seed}"

def parse_args() -> Config:
        parser = argparse.ArgumentParser()

        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--episodes", type=int, default=500)
        parser.add_argument("--embed_dim", type=int, default=128)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--data_dir", type=str, default="data_version_1")
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