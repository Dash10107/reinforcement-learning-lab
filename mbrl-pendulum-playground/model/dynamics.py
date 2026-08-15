"""
Dynamics models for MBRL Pendulum.

DynamicsModel     — single MLP: (s, a) → s'
EnsembleDynamics  — 5-model ensemble: (s, a) → (mean_s', std_s')
                    Ensemble variance = epistemic uncertainty estimate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

STATE_DIM = 3  # [cos_theta, sin_theta, angular_vel]
ACTION_DIM = 1  # [torque]
STATE_NAMES = ["cos θ", "sin θ", "θ̇ (ang. vel)"]


class DynamicsModel(nn.Module):
    """Single neural network dynamics model."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, STATE_DIM),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def predict_np(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            a = torch.FloatTensor(action).reshape(1, 1)
            return self(s, a).squeeze(0).numpy()


class EnsembleDynamics:
    """
    Bootstrap ensemble of N dynamics models.
    Provides mean prediction and uncertainty (std across ensemble members).
    """

    def __init__(self, n: int = 5, hidden: int = 128):
        self.models = nn.ModuleList([DynamicsModel(hidden) for _ in range(n)])
        self.n = n

    def parameters(self):
        return self.models.parameters()

    def train(self):
        for m in self.models:
            m.train()

    def eval(self):
        for m in self.models:
            m.eval()

    def forward_all(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([m(state, action) for m in self.models])  # (N, B, 3)
        return preds.mean(0), preds.std(0)

    def predict_np(
        self, state: np.ndarray, action: float
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            a = torch.FloatTensor([[action]])
            mean, std = self.forward_all(s, a)
            return mean.squeeze(0).numpy(), std.squeeze(0).numpy()

    def save(self, path: str):
        torch.save([m.state_dict() for m in self.models], path)

    def load(self, path: str):
        states = torch.load(path, map_location="cpu")
        for m, sd in zip(self.models, states):
            m.load_state_dict(sd)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward_all(state, action)
        return mean


def load_pretrained(path: str = "dynamics_model.pth") -> DynamicsModel:
    """Load original single-model checkpoint (backward compat)."""
    sd = torch.load(path, map_location="cpu")
    # Detect key format: original app used a DynamicsModel with self.net
    # Check if keys have 'net.' prefix
    if any(k.startswith("net.") for k in sd):
        # Original format: net.0.weight etc — build matching architecture
        class _OrigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(STATE_DIM + ACTION_DIM, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, STATE_DIM),
                )

            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                return self.net(x)

            def predict_np(self, state, action):
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0)
                    a = torch.FloatTensor(action).reshape(1, 1)
                    return self(s, a).squeeze(0).numpy()

        m = _OrigModel()
        m.load_state_dict(sd)
        m.eval()
        return m  # type: ignore[return-value]
    else:
        model = DynamicsModel(hidden=64)
        model.load_state_dict(sd)
        model.eval()
        return model


def pendulum_reward_batch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Pendulum-v1 reward function (vectorised, differentiable).
    states : (B, 3) — [cos_th, sin_th, ang_vel]
    actions: (B, 1) — torque ∈ [-2, 2]
    """
    theta = torch.atan2(states[:, 1], states[:, 0])  # recover angle
    ang_vel = states[:, 2]
    torque = actions[:, 0].clamp(-2, 2)
    return -(theta.pow(2) + 0.1 * ang_vel.pow(2) + 0.001 * torque.pow(2))
