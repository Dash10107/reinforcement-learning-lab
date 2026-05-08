"""
Data collection and dynamics model training pipeline.
Runs in a background thread; progress written to TrainingState.
"""

from __future__ import annotations
import threading
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from dataclasses import dataclass, field

from model.dynamics import EnsembleDynamics, STATE_DIM, ACTION_DIM

ENSEMBLE_PATH = "ensemble_dynamics.pt"


@dataclass
class TrainingState:
    running: bool = False
    phase: str = "idle"       # "collecting" | "training" | "done" | "idle"
    n_transitions: int = 0
    epoch: int = 0
    total_epochs: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    status: str = "No model trained yet."
    model_ready: bool = False


def collect_transitions(
    n_steps: int = 3000,
    state: TrainingState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out a random policy in Pendulum-v1 and collect (s, a, s') tuples.
    Returns arrays of shape (N, 3), (N, 1), (N, 3).
    """
    env = gym.make("Pendulum-v1")
    states, actions, next_states = [], [], []
    obs, _ = env.reset()
    step = 0

    while step < n_steps:
        action = env.action_space.sample()
        next_obs, _, term, trunc, _ = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())
        next_states.append(next_obs.copy())
        obs = next_obs
        step += 1
        if term or trunc:
            obs, _ = env.reset()
        if state:
            state.n_transitions = step

    env.close()
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32).reshape(-1, 1),
        np.array(next_states, dtype=np.float32),
    )


def train_ensemble(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_frac: float = 0.10,
    train_state: TrainingState | None = None,
    save_path: str = ENSEMBLE_PATH,
) -> EnsembleDynamics:

    ensemble = EnsembleDynamics(n=5, hidden=128)
    ensemble.train()
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Train / val split
    N = len(states)
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    S  = torch.FloatTensor(states)
    A  = torch.FloatTensor(actions)
    S_ = torch.FloatTensor(next_states)

    if train_state:
        train_state.total_epochs = epochs

    for ep in range(epochs):
        if train_state and not train_state.running:
            break

        # Shuffle train set
        perm = torch.randperm(len(train_idx))
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), batch_size):
            bi = train_idx[perm[start:start + batch_size]]
            s_b  = S[bi]
            a_b  = A[bi]
            s_b_ = S_[bi]

            optimizer.zero_grad()
            # Each model in ensemble sees a bootstrap sample
            total = torch.tensor(0.0)
            for model in ensemble.models:
                boot = torch.randint(0, len(bi), (len(bi),))
                pred = model(s_b[boot], a_b[boot])
                loss = criterion(pred, s_b_[boot])
                total = total + loss
            total.backward()
            nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
            optimizer.step()
            ep_loss += total.item() / ensemble.n
            n_batches += 1

        scheduler.step()

        # Validation
        with torch.no_grad():
            s_v  = S[val_idx]
            a_v  = A[val_idx]
            s_v_ = S_[val_idx]
            v_preds, _ = ensemble.forward_all(s_v, a_v)
            val_loss = criterion(v_preds, s_v_).item()

        train_l = ep_loss / max(n_batches, 1)

        if train_state:
            train_state.epoch = ep + 1
            train_state.train_losses.append(train_l)
            train_state.val_losses.append(val_loss)
            train_state.status = (
                f"Epoch {ep+1}/{epochs}  |  "
                f"Train MSE: {train_l:.5f}  |  Val MSE: {val_loss:.5f}"
            )

    ensemble.eval()
    ensemble.save(save_path)

    if train_state:
        train_state.running = False
        train_state.model_ready = True
        train_state.phase = "done"
        train_state.status = (
            f"Training complete!  Val MSE: {train_state.val_losses[-1]:.5f}  "
            f"({len(train_state.train_losses)} epochs)"
        )

    return ensemble


def start_training_thread(
    n_steps: int,
    epochs: int,
    train_state: TrainingState,
) -> threading.Thread:

    def _run():
        train_state.running = True
        train_state.phase = "collecting"
        train_state.status = f"Collecting {n_steps} transitions…"

        S, A, S_ = collect_transitions(n_steps=n_steps, state=train_state)

        train_state.phase = "training"
        train_state.status = "Starting model training…"

        train_ensemble(S, A, S_, epochs=epochs, train_state=train_state)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
