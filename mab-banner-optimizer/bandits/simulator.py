"""
Simulation runner — runs one or multiple agents on the same environment.
"""

from __future__ import annotations
import numpy as np
from bandits.environment import CampaignEnvironment, BannerArm
from bandits.agents import BanditAgent, make_agent


def run_single(
    env: CampaignEnvironment,
    agent: BanditAgent,
    n_steps: int,
    seed: int = 42,
) -> BanditAgent:
    """Run one agent on the environment for n_steps."""
    env.reset(seed=seed)
    for _ in range(n_steps):
        arm = agent.choose()
        reward, converted = env.pull(arm)
        agent.update(arm, reward, converted, env.optimal_ev)
    return agent


def run_comparison(
    arms: list[BannerArm],
    agent_names: list[str],
    n_steps: int,
    drift_std: float = 0.002,
    seed: int = 42,
    agent_params: dict | None = None,
) -> dict[str, BanditAgent]:
    """
    Run multiple agents on IDENTICAL environments (same seed per agent).
    Returns dict of agent_name → trained agent.
    """
    params = agent_params or {}
    results: dict[str, BanditAgent] = {}

    for name in agent_names:
        env = CampaignEnvironment(arms, drift_std=drift_std, seed=seed)
        agent = make_agent(name, len(arms), params.get(name, {}))
        run_single(env, agent, n_steps, seed=seed)
        results[name] = agent

    return results


def run_averaged(
    arms: list[BannerArm],
    agent_name: str,
    n_steps: int,
    n_runs: int = 20,
    drift_std: float = 0.002,
    agent_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average over n_runs to get smooth regret/reward curves.
    Returns (mean_regret, std_regret, mean_reward) arrays of length n_steps.
    """
    params = agent_params or {}
    all_regrets: list[list[float]] = []
    all_rewards: list[list[float]] = []

    for run in range(n_runs):
        env   = CampaignEnvironment(arms, drift_std=drift_std, seed=run)
        agent = make_agent(agent_name, len(arms), params.get(agent_name, {}))
        run_single(env, agent, n_steps, seed=run)
        all_regrets.append(agent.regret_history[:n_steps])
        all_rewards.append(list(np.cumsum(agent.reward_history[:n_steps])))

    R = np.array(all_regrets)
    W = np.array(all_rewards)
    return R.mean(0), R.std(0), W.mean(0)
