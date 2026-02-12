"""
Ad campaign environment — simulates banner impressions with stochastic rewards.
Each arm = one banner variant with its own CTR and revenue-per-conversion.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BannerArm:
    name: str
    true_ctr: float          # true click-through rate (hidden from agents)
    revenue: float           # revenue per conversion ($)
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    total_revenue: float = 0.0

    @property
    def observed_ctr(self) -> float:
        return self.clicks / max(self.impressions, 1)

    @property
    def expected_value(self) -> float:
        return self.true_ctr * self.revenue


SCENARIOS = {
    "🛒 E-Commerce Sale": {
        "description": "4 banner variants for a flash sale — different designs, same product",
        "arms": [
            BannerArm("Red Urgency",    true_ctr=0.04, revenue=45.0),
            BannerArm("Blue Minimal",   true_ctr=0.09, revenue=45.0),
            BannerArm("Green Social",   true_ctr=0.06, revenue=45.0),
            BannerArm("Video Teaser",   true_ctr=0.12, revenue=45.0),
        ],
    },
    "📰 News Feed CTR": {
        "description": "Headlines A/B/C/D — optimise for ad clicks",
        "arms": [
            BannerArm("Curiosity Gap",  true_ctr=0.08, revenue=1.2),
            BannerArm("Direct Value",   true_ctr=0.05, revenue=1.2),
            BannerArm("Social Proof",   true_ctr=0.11, revenue=1.2),
            BannerArm("Listicle",       true_ctr=0.07, revenue=1.2),
            BannerArm("Fear Of Loss",   true_ctr=0.03, revenue=1.2),
        ],
    },
    "💰 SaaS Pricing Page": {
        "description": "3 CTA variants — optimise for trial sign-ups",
        "arms": [
            BannerArm("Free Trial",     true_ctr=0.15, revenue=120.0),
            BannerArm("Demo Request",   true_ctr=0.07, revenue=180.0),
            BannerArm("Buy Now",        true_ctr=0.03, revenue=300.0),
        ],
    },
    "🎮 Mobile Game Install": {
        "description": "6 creative variants for app install campaign",
        "arms": [
            BannerArm("Gameplay GIF",   true_ctr=0.05, revenue=3.5),
            BannerArm("Reward Lure",    true_ctr=0.08, revenue=3.5),
            BannerArm("Social Proof",   true_ctr=0.06, revenue=3.5),
            BannerArm("Tutorial Clip",  true_ctr=0.04, revenue=3.5),
            BannerArm("Challenge CTA",  true_ctr=0.10, revenue=3.5),
            BannerArm("High Score",     true_ctr=0.07, revenue=3.5),
        ],
    },
    "🔬 Custom": {
        "description": "Build your own scenario",
        "arms": [],
    },
}


class CampaignEnvironment:
    """
    Stochastic non-stationary ad environment.

    Each pull of arm i:
      - Bernoulli(true_ctr[i]) determines if conversion happens
      - Reward = revenue[i] if converted else 0
      - Optional drift: CTRs slowly change over time (real-world non-stationarity)
    """

    def __init__(
        self,
        arms: list[BannerArm],
        drift_std: float = 0.002,
        seed: int | None = None,
    ):
        self.arms = [BannerArm(**vars(a)) for a in arms]  # deep copy
        self.drift_std = drift_std
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self._true_ctrs = np.array([a.true_ctr for a in self.arms])

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax([a.expected_value for a in self.arms]))

    @property
    def optimal_ev(self) -> float:
        return max(a.expected_value for a in self.arms)

    def pull(self, arm_idx: int) -> tuple[float, bool]:
        """Pull arm, return (reward, converted)."""
        arm = self.arms[arm_idx]
        converted = bool(self.rng.random() < arm.true_ctr)
        reward = arm.revenue * converted

        arm.impressions += 1
        if converted:
            arm.clicks += 1
            arm.conversions += 1
            arm.total_revenue += reward

        # Drift
        if self.drift_std > 0:
            noise = self.rng.normal(0, self.drift_std, self.n_arms)
            new_ctrs = self._true_ctrs + noise
            self._true_ctrs = np.clip(new_ctrs, 0.01, 0.95)
            for i, arm in enumerate(self.arms):
                arm.true_ctr = float(self._true_ctrs[i])

        self.t += 1
        return reward, converted

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for i, arm in enumerate(self.arms):
            arm.true_ctr = float(self._true_ctrs[i])
            arm.impressions = arm.clicks = arm.conversions = 0
            arm.total_revenue = 0.0
        self.t = 0
