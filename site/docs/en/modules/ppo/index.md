---
description: "PPO (Proximal Policy Optimization) explained — the algorithm behind ChatGPT, OpenAI Five, and robotics. Covers clipping trick, full PPO objective with math, and comparison to TRPO."
---

# PPO Explained — Proximal Policy Optimization Tutorial
<br> *The algorithm behind modern AI.*

PPO is behind most of the biggest RL results in the last five years.

It trained the robots that OpenAI's Dactyl used to manipulate a Rubik's cube with a single hand. It ran the bots that beat professional players at Dota 2. It's the RL component inside InstructGPT — the system that aligned ChatGPT with human preferences.

It's also surprisingly simple to understand. The core idea fits in one sentence:

**Don't change the policy too much in a single update.**

---

## The problem PPO solves

In Actor-Critic, there's nothing stopping the policy gradient update from making a very large change. One bad batch of experiences, one large gradient step, and the policy can flip completely — from "usually go left" to "always go right." The Critic's value estimates, trained for the old policy, are now wrong. The next update is based on wrong estimates. Things spiral.

This is called **catastrophic forgetting in RL**: the policy updates so aggressively that it loses what it learned before.

PPO's solution: add a constraint that limits how far the new policy can deviate from the old one.

---

## The clipping trick

PPO measures how much the policy has changed using a ratio:

```
ratio = π_new(action | state) / π_old(action | state)
```

If the ratio is 1.0, the policy hasn't changed. If it's 2.0, the new policy is twice as likely to take that action as the old one. If it's 0.5, it's half as likely.

PPO clips this ratio to stay between `1 - ε` and `1 + ε` (where ε is typically 0.2):

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    # How much has the policy changed for each action?
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Unclipped loss: standard policy gradient
    unclipped = ratio * advantages

    # Clipped loss: ratio can't move outside [1-ε, 1+ε]
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    # Take the more pessimistic of the two
    return -torch.min(unclipped, clipped).mean()
```

The key line is `torch.min(unclipped, clipped)`. We always take the *worse* of the two estimates. This means:

- If an action had a **positive** advantage (do more of it): we allow ratio to increase, but only up to `1 + ε`. Beyond that, we get no extra credit.
- If an action had a **negative** advantage (do less of it): we allow ratio to decrease, but only down to `1 - ε`. Beyond that, no extra penalty.

The clipping creates a "trust region" — a safe zone where the gradient update can move freely. Outside the zone, updates are cut off. The policy can only change so much at once.

---

## The full PPO objective

PPO optimises a combined loss with three terms:

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \mathcal{L}_t^{\text{CLIP}} - c_1 \mathcal{L}_t^{\text{VF}} + c_2 H[\pi_\theta](s_t) \right]$$

Where:
- $\mathcal{L}_t^{\text{CLIP}} = \min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1{-}\varepsilon, 1{+}\varepsilon)\hat{A}_t\right)$ — the clipped policy gradient
- $\mathcal{L}_t^{\text{VF}} = (V_\theta(s_t) - V_t^{\text{target}})^2$ — the critic's value function loss
- $H[\pi_\theta](s_t)$ — the policy entropy at state $s_t$
- $c_1 \approx 0.5$, $c_2 \approx 0.01$ — coefficients balancing the three terms

In plain English: *improve the policy (CLIP term), make the value estimates accurate (VF term), and keep the policy from collapsing too early (entropy term) — all in one pass.*

The entropy term is the same bonus we saw in A2C. It prevents the policy from becoming deterministic too quickly. Without it, PPO can sometimes "commit" to a suboptimal action and never explore alternatives.

---

## Where this came from: TRPO

PPO's predecessor was **TRPO** (Trust Region Policy Optimization), which added a formal constraint on how much the policy could change:

$$\mathbb{E}\left[\text{KL}[\pi_{\theta_{\text{old}}} \| \pi_\theta]\right] \leq \delta$$

In plain English: *the KL divergence between old and new policy must stay below a threshold $\delta$.* KL divergence measures how different two probability distributions are. The constraint forces the new policy to be "close" to the old one in a probabilistic sense.

TRPO is theoretically clean but computationally expensive — enforcing the constraint requires computing second-order derivatives (the natural gradient). PPO's clipping approximates the same effect with only first-order derivatives (standard backpropagation). That's why PPO runs on GPUs and TRPO mostly doesn't.

---

## Why this is enough

PPO doesn't add a hard constraint (which would require solving a constrained optimisation problem). It just clips.

This is much simpler computationally — it's just a `min` operation. But it achieves a similar effect: the policy is forced to move slowly. Updates are stable. The Critic's estimates stay valid. Training doesn't spiral.

The elegance of PPO is that a very simple mechanical trick has a very large practical effect.

---

## Where PPO is running in the real world

**Game AI.** OpenAI Five (Dota 2) ran 180 years of self-play per day using PPO. The bots learned complex team strategies — warding, baiting, counter-picks — from scratch, just by playing the game.

**Robotics.** OpenAI Dactyl solved a Rubik's cube with a robot hand using PPO with domain randomisation (training across thousands of different simulated physics conditions to make the policy robust to the real world).

**Language models.** InstructGPT — the system that became the foundation for ChatGPT — used PPO as the RL component in RLHF. A reward model rated responses based on human preferences; PPO trained the language model to maximise those ratings.

PPO doesn't just train agents in games. It's one of the tools shaping how AI systems behave.

---

## What you'll notice in the demo

Open the [Unity 3D Dog Walker ↗](https://huggingface.co/spaces/Dash10107/Unity-RL-Huggy-Demo) — a PPO agent learning to walk and fetch.

**Three things to watch:**

1. **Early locomotion.** The dog falls constantly. Its joints are firing randomly. This is not failure — it's the initial random policy. Reward is near zero.

2. **Gait emergence.** Around episode 1,000, something clicks. A consistent gait emerges — not the optimal gait, but something stable. PPO found a local structure that generates reward reliably.

3. **Policy smoothness.** Compare early and late policy outputs. Early: large, erratic joint angles. Late: smooth, coordinated movements. PPO's conservative updates accumulate into a refined policy.

---

## Try it yourself

**Experiment 1 — Increase epsilon.**
Set ε to 0.5 (allow larger updates). Early training might go faster — but watch for instability. The policy can jump too far and collapse.

**Experiment 2 — Decrease epsilon.**
Set ε to 0.05 (very small updates only). Training becomes very slow. The policy is being overly cautious — it barely changes from episode to episode. Lower epsilon = more stable but slower.

**Experiment 3 — Number of epochs per rollout.**
PPO reuses each batch of experience for multiple gradient updates (this is what makes it sample-efficient). Increase the number of epochs from 4 to 20. The network will start overfitting to recent experience — you'll see the surrogate loss diverge.

---

## One limitation

PPO still outputs action *probabilities* over a discrete action set. For most robotics and control tasks, the action is not "pick from a menu" — it's a precise number. What angle should this joint be at? How much force should this actuator apply?

PPO can be extended to continuous actions using a Gaussian distribution, but in practice there's an algorithm that handles continuous control more naturally — and handles exploration more cleverly.

That's SAC.
