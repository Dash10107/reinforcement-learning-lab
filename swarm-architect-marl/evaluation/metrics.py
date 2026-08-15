from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingLog:
    episodes: list[int] = field(default_factory=list)
    mean_rewards: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    _recent: deque = field(default_factory=lambda: deque(maxlen=50))

    def record(self, episode: int, rewards: dict, metrics: dict):
        mean_r = np.mean(list(rewards.values()))
        self._recent.append(mean_r)
        self.episodes.append(episode)
        self.mean_rewards.append(mean_r)

        agent_metrics = list(metrics.values())
        self.actor_losses.append(np.mean([m["actor_loss"] for m in agent_metrics]))
        self.critic_losses.append(np.mean([m["critic_loss"] for m in agent_metrics]))
        self.entropies.append(np.mean([m["entropy"] for m in agent_metrics]))

    @property
    def rolling_mean(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0

    @property
    def best_reward(self) -> float:
        return max(self.mean_rewards) if self.mean_rewards else 0.0

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "mean_rewards": self.mean_rewards,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "entropies": self.entropies,
        }


def evaluate_agents(env, agents: dict, n_episodes: int = 5) -> dict:
    total_rewards = {aid: 0.0 for aid in agents}
    episode_lengths = []

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        ep_len = 0
        ep_rewards = {aid: 0.0 for aid in agents}

        done = False
        while not done:
            actions = {}
            for aid, agent in agents.items():
                import numpy as np

                obs = obs_dict.get(aid, np.zeros(agent.actor.net[0].in_features))
                actions[aid] = agent.greedy_action(obs)

            obs_dict, rewards, terms, truncs, _ = env.step(actions)
            for aid in agents:
                ep_rewards[aid] += rewards.get(aid, 0.0)
            ep_len += 1
            done = all(terms.get(a, False) or truncs.get(a, False) for a in agents)

        for aid in agents:
            total_rewards[aid] += ep_rewards[aid]
        episode_lengths.append(ep_len)

    return {
        "mean_reward_per_agent": {
            aid: v / n_episodes for aid, v in total_rewards.items()
        },
        "mean_episode_length": np.mean(episode_lengths),
        "team_reward": sum(total_rewards.values()) / n_episodes,
    }
