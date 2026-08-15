import torch
from torch import nn


def _make_mlp(
    in_dim: int, hidden: list[int], out_dim: int, activation: nn.Module
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Stochastic discrete actor — outputs categorical action distribution."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int] = [128, 128]):  # noqa: B006
        super().__init__()
        self.net = _make_mlp(obs_dim, hidden, act_dim, nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self(obs)
        return dist.log_prob(action), dist.entropy()


class Critic(nn.Module):
    """Value function baseline."""

    def __init__(self, obs_dim: int, hidden: list[int] = [128, 128]):  # noqa: B006
        super().__init__()
        self.net = _make_mlp(obs_dim, hidden, 1, nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Legacy-compatible greedy policy for loading old checkpoints."""

    def __init__(self, obs_shape: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
