import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import wandb

from configs.config  import parse_args
from core.envs.multigraph_env import MultiGraphEnv
from core.models.multi_agent import MultiAgentREINFORCE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=vars(cfg)
        )

    env = MultiGraphEnv(cfg)
    agent = MultiAgentREINFORCE(cfg)

    rewards = []

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for ep in range(cfg.episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0

        # TRAINING rollout (NO no_grad)
        while not (terminated or truncated):
            actions = agent.act(state, eval_mode=False)
            state, reward, terminated, truncated, _ = env.step(actions)
            agent.store_reward(reward)
            ep_reward += reward

        loss = agent.update()
        rewards.append(ep_reward)

        if cfg.wandb:
            wandb.log({"episode": ep, "reward": ep_reward, "loss": loss})

        if (ep + 1) % cfg.log_interval == 0:
            print(f"Episode {ep+1}/{cfg.episodes} | Reward: {ep_reward:.3f} | Loss: {loss:.4f}")

    # -----------------------------
    # EVALUATION LOOP (AFTER TRAINING)
    # -----------------------------
    eval_env = MultiGraphEnv(cfg)
    state, _ = eval_env.reset()
    terminated = False
    truncated = False
    eval_reward = 0

    with torch.no_grad():
        while not (terminated or truncated):
            actions = agent.act(state, eval_mode=True)  # GREEDY
            state, reward, terminated, truncated, _ = eval_env.step(actions)
            eval_reward += reward

    print(f"\nFinal Evaluation Reward: {eval_reward}")

    if cfg.wandb:
        wandb.log({"final_eval_reward": eval_reward})

    # -----------------------------
    # PLOT TRAINING REWARD CURVE
    # -----------------------------
    plt.plot(rewards)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    if cfg.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()