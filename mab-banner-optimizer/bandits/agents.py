"""
Multi-Armed Bandit agents — all implement the same interface:
  choose(t) → arm_index
  update(arm, reward, converted) → None
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class BanditAgent(ABC):
    def __init__(self, n_arms: int, name: str):
        self.n_arms = n_arms
        self.name = name
        self.t = 0
        self.counts = np.zeros(n_arms, dtype=int)    # pulls per arm
        self.values = np.zeros(n_arms)               # estimated value per arm
        self.total_reward = 0.0
        self.regret_history: list[float] = []
        self.reward_history: list[float] = []
        self.arm_history: list[int] = []
        self.cumulative_regret = 0.0

    @abstractmethod
    def choose(self) -> int:
        ...

    def update(self, arm: int, reward: float, _converted: bool, optimal_ev: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.total_reward += reward
        regret = optimal_ev - reward
        self.cumulative_regret += regret
        self.regret_history.append(self.cumulative_regret)
        self.reward_history.append(reward)
        self.arm_history.append(arm)
        self.t += 1

    @property
    def cumulative_rewards(self) -> np.ndarray:
        return np.cumsum(self.reward_history)

    @property
    def arm_pull_pcts(self) -> np.ndarray:
        total = self.counts.sum()
        return self.counts / max(total, 1) * 100


# ── Epsilon-Greedy ────────────────────────────────────────────────────────────

class EpsilonGreedy(BanditAgent):
    """
    With probability ε: random exploration.
    Otherwise: exploit the arm with highest estimated value.
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms, f"ε-Greedy (ε={epsilon})")
        self.epsilon = epsilon

    def choose(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))


# ── Decaying Epsilon-Greedy ────────────────────────────────────────────────

class DecayingEpsilonGreedy(BanditAgent):
    """Epsilon decays as 1/sqrt(t) — more explore early, more exploit later."""
    def __init__(self, n_arms: int, decay: float = 1.0):
        super().__init__(n_arms, "Decaying ε-Greedy")
        self.decay = decay

    def choose(self) -> int:
        eps = min(1.0, self.decay / max(1, np.sqrt(self.t + 1)))
        if np.random.random() < eps:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))


# ── UCB1 ──────────────────────────────────────────────────────────────────────

class UCB1(BanditAgent):
    """
    Upper Confidence Bound: choose arm with highest Q + bonus.
    Bonus = c * sqrt(log(t) / n_pulls)
    """
    def __init__(self, n_arms: int, c: float = 2.0):
        super().__init__(n_arms, f"UCB1 (c={c})")
        self.c = c

    def choose(self) -> int:
        # Force one pull per arm first
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        t = self.t + 1
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(self.values + bonus))


# ── Thompson Sampling ─────────────────────────────────────────────────────────

class ThompsonSampling(BanditAgent):
    """
    Bayesian bandit: maintain Beta(α, β) posterior for each arm's CTR.
    Sample from posteriors, pick highest sample.
    """
    def __init__(self, n_arms: int):
        super().__init__(n_arms, "Thompson Sampling")
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta  = np.ones(n_arms)   # failures  + 1

    def choose(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, converted: bool, optimal_ev: float):
        if converted:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        # Use scaled reward for value tracking (revenue-weighted)
        super().update(arm, reward, converted, optimal_ev)

    @property
    def posterior_means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_stds(self) -> np.ndarray:
        a, b = self.alpha, self.beta
        n = a + b
        return np.sqrt(a * b / (n * n * (n + 1)))


# ── Gradient Bandit ───────────────────────────────────────────────────────────

class GradientBandit(BanditAgent):
    """
    Learns a preference H(a) for each arm using stochastic gradient ascent.
    Action probabilities: softmax(H).
    """
    def __init__(self, n_arms: int, alpha: float = 0.1):
        super().__init__(n_arms, f"Gradient Bandit (α={alpha})")
        self.alpha_lr = alpha
        self.H = np.zeros(n_arms)   # preferences
        self.baseline = 0.0          # running average reward

    def _softmax(self) -> np.ndarray:
        h = self.H - self.H.max()
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def choose(self) -> int:
        probs = self._softmax()
        return int(np.random.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float, converted: bool, optimal_ev: float):
        probs = self._softmax()
        # Gradient update
        self.H -= self.alpha_lr * (reward - self.baseline) * probs
        self.H[arm] += self.alpha_lr * (reward - self.baseline)
        # Update baseline
        n = self.t + 1
        self.baseline += (reward - self.baseline) / n
        super().update(arm, reward, converted, optimal_ev)


# ── EXP3 (Adversarial Bandit) ─────────────────────────────────────────────────

class EXP3(BanditAgent):
    """
    EXP3: Exponential-weight algorithm for Exploration and Exploitation.
    Works even in adversarial (non-stochastic) environments.
    """
    def __init__(self, n_arms: int, gamma: float = 0.1):
        super().__init__(n_arms, f"EXP3 (γ={gamma})")
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    def _probs(self) -> np.ndarray:
        w = self.weights
        uniform = np.ones(self.n_arms) / self.n_arms
        return (1 - self.gamma) * w / w.sum() + self.gamma * uniform

    def choose(self) -> int:
        probs = self._probs()
        return int(np.random.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float, converted: bool, optimal_ev: float):
        probs = self._probs()
        # Importance-weighted update (normalise reward to [0,1])
        max_r = max(r for a in [arm] for r in [reward + 1e-9])
        r_hat = reward / (max_r * probs[arm] + 1e-9)
        self.weights[arm] *= np.exp(self.gamma * r_hat / self.n_arms)
        super().update(arm, reward, converted, optimal_ev)


# ── Factory ───────────────────────────────────────────────────────────────────

AGENT_REGISTRY = {
    "Thompson Sampling":     lambda n, p: ThompsonSampling(n),
    "UCB1":                  lambda n, p: UCB1(n, c=p.get("c", 2.0)),
    "Epsilon-Greedy":        lambda n, p: EpsilonGreedy(n, epsilon=p.get("epsilon", 0.1)),
    "Decaying Epsilon-Greedy": lambda n, p: DecayingEpsilonGreedy(n),
    "Gradient Bandit":       lambda n, p: GradientBandit(n, alpha=p.get("alpha", 0.1)),
    "EXP3 (Adversarial)":    lambda n, p: EXP3(n, gamma=p.get("gamma", 0.1)),
}

AGENT_COLORS = {
    "Thompson Sampling":     "#00d4aa",
    "UCB1":                  "#1890ff",
    "Epsilon-Greedy":        "#faad14",
    "Decaying Epsilon-Greedy": "#f59e0b",
    "Gradient Bandit":       "#a855f7",
    "EXP3 (Adversarial)":    "#ef4444",
}


def make_agent(name: str, n_arms: int, params: dict | None = None) -> BanditAgent:
    return AGENT_REGISTRY[name](n_arms, params or {})
