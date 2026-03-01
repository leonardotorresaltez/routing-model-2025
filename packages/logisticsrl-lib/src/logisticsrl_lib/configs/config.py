import argparse
from dataclasses import dataclass

@dataclass
class Config:
    # --- Experiment ---
    project_name: str = "routing-model-2025"
    run_name: str = "default"
    seed: int = 42
    device: str = "cpu" # Tip: switch to 'cuda' or 'mps' if available!
    wandb: bool = True  
    data_dir: str = "data_version_1"  
    
    # --- Model ---
    embed_dim: int = 2**6 
    max_daily_delivery_time_each_truck: int = 24  
    
    # --- Training (Optimized for PPO) ---
    lr: float = 1e-3                 # Increased for PPO (standard is 3e-4)
    episodes: int = int(1e5)            
    log_interval: int = 20
    # gamma: float = 0.99         
    gamma: float = 1         
    # reward_scale: float = 1/150  
    returns_var_alpha: float = 1/200  # Smoothing factor for EMA of returns variance (for normalization). Approx half-life of 200 batches.
    distance_penalty_scale: float = 1/5   # Scale for the distance penalty in the reward function (to keep rewards in a reasonable range for PPO)
    
    # --- PPO Specifics ---
    episodes_per_update_batch: int = 10
    ppo_epochs: int = 4              # How many times to loop over the batch
    eps_clip: float = 0.1            # PPO clipping ratio
    value_coef: float = 0.5          # How much the Critic loss matters
    entropy_bonus: float = 0.01      # typical value around 0.01

    
    max_constant_routes =  episodes_per_update_batch * 2 + 2  

    # --- Debug ---
    debug: bool = False  # Set to True for quick testing with minimal episodes and smaller model
    if debug:
        episodes = 10
        episodes_per_update_batch = 1
        wandb = False

    def __post_init__(self):
        if self.data_dir == "data_version_2":
            self.max_daily_delivery_time_each_truck = 24
            
        else:
            self.max_daily_delivery_time_each_truck = 12
        
        # Construct project_name
        self.project_name = f"{self.project_name}_{self.data_dir}"
        # Construct Run Name
        # self.run_name = f"lr{self.lr}_gamma{self.gamma}_sd{self.seed}"

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
    parser.add_argument("--run_name", type=str, default=base_cfg.run_name)
    
    # Flag: --no-wandb to disable logging
    # parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    
    return Config(
        lr=args.lr,
        episodes=args.episodes,
        embed_dim=args.embed_dim,
        seed=args.seed,
        device=args.device,
        # wandb=not args.no_wandb,
        data_dir=args.data_dir,
        run_name=args.run_name,
    )