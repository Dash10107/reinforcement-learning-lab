"""
AI Tutor RL Environment — models a student learning across N subjects.

State:  proficiency scores in [0, 1] for each subject
Action: which subject to focus on (0..N-1)
Reward: proficiency of the focused subject after the step
  → encourages the agent to focus on subjects where it can make progress

Dynamics:
  - Focused subject gains +15–30% (simulates a focused study session)
  - All other subjects decay -1–3% (simulates forgetting / drift)
  - Episode ends when all subjects reach 98%+ mastery
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SUBJECTS = [
    "Mathematics",
    "Physics",
    "Literature",
    "History",
    "Computer Science",
]
SUBJECT_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#3b82f6"]
SUBJECT_ICONS = ["∑", "⚛", "📖", "🏛", "</>"]

N_SUBJECTS = len(SUBJECTS)


class AITutorEnv(gym.Env):
    """
    Gymnasium-compatible tutoring environment.

    Observation space: Box([0,1]^N) — normalised proficiency per subject
    Action space:      Discrete(N)  — which subject to study
    """

    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(self, n_subjects: int = N_SUBJECTS):
        super().__init__()
        self.n = n_subjects
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_subjects,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_subjects)
        self.state = np.zeros(n_subjects, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 200  # hard cap prevents infinite episodes

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Start students with low, varied proficiency
        self.state = np.random.uniform(0.05, 0.35, self.n).astype(np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action: int):
        self.step_count += 1

        # Learning gain for focused subject
        gain = float(np.random.uniform(0.12, 0.28))
        self.state[action] = min(1.0, self.state[action] + gain)

        # Forgetting for all other subjects
        decay = np.random.uniform(0.005, 0.025, self.n)
        decay[action] = 0.0
        self.state = np.maximum(0.0, self.state - decay)

        reward = float(self.state[action])  # reward = current mastery
        mastered = bool(np.all(self.state >= 0.98))
        truncated = self.step_count >= self.max_steps
        done = mastered

        info = {"mastered": mastered, "step": self.step_count}
        return self.state.copy(), reward, done, truncated, info

    def set_state(self, proficiency_pct: list[float]):
        """Inject a specific state (from UI sliders, in 0–100 range)."""
        self.state = np.clip(
            np.array(proficiency_pct, dtype=np.float32) / 100.0, 0.0, 1.0
        )
        self.step_count = 0
