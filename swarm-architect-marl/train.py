"""
CLI training script — run this standalone to train agents:
  python train.py
  python train.py --episodes 1000 --lr 0.0003
"""

import argparse
import os
import time

from config import ENV, PPO, NET, TRAIN
from agents.ippo import IPPOTrainer
from environment.wrapper import make_env, get_agent_ids
from evaluation.metrics import TrainingLog, evaluate_agents


def train(
    total_episodes: int = TRAIN.total_episodes,
    rollout_steps: int = PPO.rollout_steps,
    lr_actor: float = PPO.lr_actor,
    checkpoint_dir: str = TRAIN.checkpoint_dir,
    verbose: bool = True,
):
    PPO.rollout_steps = rollout_steps
    PPO.lr_actor = lr_actor

    train_env = make_env(ENV, render=False)
    eval_env = make_env(ENV, render=False)

    agent_ids = get_agent_ids(train_env)
    trainer = IPPOTrainer(ENV, PPO, NET)
    trainer.init_agents(agent_ids)

    # Try to resume from best checkpoint
    best_path = os.path.join(checkpoint_dir, "best_agent.pt")
    if trainer.load_best(best_path):
        print(f"Resumed from {best_path}")

    log = TrainingLog()
    best_reward = float("-inf")
    start = time.time()

    for ep in range(1, total_episodes + 1):
        rewards, metrics = trainer.train_episode(train_env)
        log.record(ep, rewards, metrics)

        if ep % TRAIN.log_every == 0 and verbose:
            elapsed = time.time() - start
            print(
                f"Ep {ep:4d}/{total_episodes} | "
                f"Reward: {log.mean_rewards[-1]:+.4f} | "
                f"Rolling: {log.rolling_mean:+.4f} | "
                f"ActorLoss: {log.actor_losses[-1]:.4f} | "
                f"Entropy: {log.entropies[-1]:.4f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

        if ep % TRAIN.save_every == 0:
            trainer.save_all(checkpoint_dir, ep)

        if log.rolling_mean > best_reward:
            best_reward = log.rolling_mean
            trainer.save_all(checkpoint_dir, 0)  # ep=0 → best slot
            if verbose:
                print(f"  *** New best rolling reward: {best_reward:+.4f} ***")

    train_env.close()
    eval_env.close()
    print(f"\nTraining complete. Best reward: {best_reward:+.4f}")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TRAIN.total_episodes)
    parser.add_argument("--rollout", type=int, default=PPO.rollout_steps)
    parser.add_argument("--lr", type=float, default=PPO.lr_actor)
    parser.add_argument("--checkpoint_dir", type=str, default=TRAIN.checkpoint_dir)
    args = parser.parse_args()

    train(
        total_episodes=args.episodes,
        rollout_steps=args.rollout,
        lr_actor=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
