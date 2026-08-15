from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(obs))

    def act(self, obs: torch.Tensor):
        dist = self(obs)
        a = dist.sample()
        return a, dist.log_prob(a)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self(obs)
        return dist.log_prob(actions), dist.entropy()

    def greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            return int(self(t).probs.argmax().item())


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
