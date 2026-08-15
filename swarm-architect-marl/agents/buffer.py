from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Rollout:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def append(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float = 0.95,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values + [last_value], dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


def build_minibatches(
    obs: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    batch_size: int,
):
    n = obs.shape[0]
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield obs[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx]
