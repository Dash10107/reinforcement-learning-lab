---
description: "Rocket Lander demo — watch SAC control continuous engine thrust to land a rocket. See maximum entropy exploration and soft policy updates in action."
---

# 🚀 Rocket Lander

A SAC (Soft Actor-Critic) agent controls the thrust of a rocket's engines in real time, learning to land from any starting position. The agent's only input: position, velocity, angle, and angular velocity. No physics equations hardcoded.

<a href="https://huggingface.co/spaces/Dash10107/rocket-lander-sac" target="_blank" rel="noopener" class="hero-btn-primary" style="display:inline-flex;margin-bottom:1.5rem">
  ▶ Open Live Demo ↗
</a>

---

## What this demo shows

**Continuous action space.** Unlike the maze (4 discrete directions), the rocket controls a continuous thrust value from 0.0 to 1.0. SAC outputs a Gaussian distribution over thrust — it samples from that distribution to act. The distribution gets narrower as training progresses.

**The entropy curve.** Watch the `α × H(π)` metric over training. It doesn't go to zero — it stabilises at a healthy minimum. This is SAC's defining feature: the policy stays slightly random by design. This randomness is why the agent handles wind gusts — it never fully committed to a single thrust profile.

**The twin critics.** SAC uses two Q-networks and takes the minimum of their estimates. This prevents overestimation. Watch both critic losses — they track each other closely but never agree exactly. The minimum-of-two is what keeps the policy conservative.

**Polyak averaging.** The target networks update via exponential moving average (`τ = 0.005`). Changes are tiny and smooth — no abrupt jumps that would destabilise training.

---

## Try these experiments

**Experiment 1 — Turbulence test.**
Let the agent land cleanly a few times. Then enable wind. Watch the first few attempts fail — then watch the agent adapt. Because SAC's policy retains entropy, it adjusts rather than breaking. A deterministic policy would require retraining.

**Experiment 2 — Crash the entropy.**
Set `α = 0.0` (no entropy bonus). The agent becomes deterministic faster, but compare performance under perturbation. Without the entropy regularisation, the policy has less room to adapt.

**Experiment 3 — Watch the action distribution.**
In the "Actor" panel, the action distribution histogram shows how confident the agent is. Early: wide bell curve. Mid-training: narrowing. Late: a sharp spike centred near the optimal thrust. Notice it never collapses to a Dirac delta — SAC won't let it.

---

## The numbers behind this demo

| Hyperparameter | Value | Why |
|---------------|-------|-----|
| Buffer size | 1,000,000 | Enough history to decorrelate samples |
| Batch size | 256 | Large enough for stable gradient estimates |
| Polyak τ | 0.005 | Slow target network updates = stable targets |
| Discount γ | 0.99 | Cares about rewards up to ~100 steps ahead |
| Auto-α | Yes | Temperature adjusts to maintain target entropy |
| Twin critics | Yes | Prevents Q-value overestimation |

---

## The chapter behind this demo

- **[Continuous Control (SAC)](../../modules/sac/)** — covers the full SAC objective, maximum entropy RL, twin critics, reparameterisation trick, and Polyak averaging

---

**Difficulty:** Advanced · **Algorithm:** SAC · **Action space:** Continuous
