from __future__ import annotations
import numpy as np
from agents.base import TabularAgent
from maze.env import MazeEnv


def train_montecarlo(
    env: MazeEnv, episodes: int, alpha: float, gamma: float,
    decay: float, seed: int = 0,
) -> tuple[TabularAgent, list[float]]:
    agent = TabularAgent(env.n_states, env.action_space.n, alpha, gamma)
    rng = np.random.default_rng(seed)
    returns_sum = np.zeros_like(agent.Q)
    returns_cnt = np.zeros_like(agent.Q)
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode: list[tuple[int, int, float]] = []
        for _ in range(env.n_states * 4):
            action = agent.choose_action(state, rng)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        # First-visit MC update
        G = 0.0
        visited: set[tuple[int, int]] = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[s, a] += G
                returns_cnt[s, a] += 1
                agent.Q[s, a] = returns_sum[s, a] / returns_cnt[s, a]

        agent.decay_epsilon(decay)
        rewards.append(sum(r for _, _, r in episode))

    return agent, rewards
