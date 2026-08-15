---
description: "Soft Actor-Critic (SAC) explained for continuous control. Covers maximum entropy objective, Gaussian policy, twin critics, Polyak averaging, and the reparameterisation trick."
---

# When Actions Are a Number, Not a Button

A robot arm doesn't choose from a list. It outputs a precise angle — somewhere between -180 and 180 degrees. A rocket thruster doesn't pick "fire" or "don't fire." It decides how much thrust, from 0% to 100%.

These are **continuous action spaces**. And they break DQN and standard policy gradient methods in a fundamental way.

With DQN, you take the maximum Q-value across all actions. If there are infinitely many possible actions — every real number in a range — you can't enumerate them. There is no `argmax` over a continuous set.

With basic policy gradients, you output probabilities over a discrete list. There's no discrete list here.

SAC — Soft Actor-Critic — is built from the ground up to handle continuous actions. And it adds one idea that makes it more capable than most discrete-action algorithms: it deliberately keeps some randomness in its policy, even after training.

---

## The Gaussian actor

Instead of outputting probabilities over a list, the SAC actor outputs the **mean and standard deviation of a Gaussian distribution** — a bell curve.

```python
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        features = self.net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # differentiable sample (reparameterisation trick)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Squash to [-1, 1] using tanh
        action = torch.tanh(action)
        # Correct log_prob for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob
```

At each step, the actor samples an action from this bell curve. Early in training, the standard deviation is high — the bell curve is wide, the agent explores freely. As training progresses, the bell curve narrows around the good actions, and behaviour becomes more precise.

---

## Maximum entropy: why staying random is smart

Here's the idea that sets SAC apart.

Most RL algorithms try to find *the* optimal policy — one specific behaviour that maximises reward. SAC tries to find a policy that maximises reward *and* stays as random as possible.

This sounds backwards. But it has a real benefit: a policy that's still somewhat random is more robust. It hasn't committed to a single approach, so when the environment changes slightly — a gust of wind, a different starting position — the policy can adapt rather than break.

The SAC objective is:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t \left( r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right) \right]$$

Where:
- $r(s_t, a_t)$ = the environment reward at each step
- $H(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s_t)]$ = the **entropy** of the policy — how random it is at state $s_t$
- $\alpha$ = the **temperature** parameter, controlling how much entropy is worth
- The agent is rewarded for both collecting reward *and* being uncertain

In plain English: *the agent is penalised for becoming too confident. Every step, it gets the environment reward plus a bonus proportional to how random its policy still is.*

$\alpha$ is learned automatically. SAC continuously adjusts it to maintain a target level of entropy — if the policy becomes too deterministic, $\alpha$ increases to push it back toward exploration. If it's too random, $\alpha$ decreases. This is the key stability advantage of SAC over manual entropy tuning.

---

## Twin critics: two pessimists are better than one optimist

SAC uses two critic networks instead of one. Both estimate the Q-value. When making decisions, SAC uses the *lower* of the two estimates.

```python
# Use the more pessimistic Q-value estimate
q1 = critic1(state, action)
q2 = critic2(state, action)
q_target = torch.min(q1, q2)  # pessimistic: don't overestimate
```

This prevents a common failure mode called **overestimation bias**: a single critic tends to overestimate Q-values, which makes the actor chase illusory rewards. With two critics and a `min`, the estimates are more conservative and more accurate.

---

## Polyak averaging: slow and steady

The target networks in SAC don't update by copying weights every N steps. They update *continuously*, by slowly blending toward the online network:

```python
# Soft update: move target a tiny fraction toward online network
tau = 0.005
for online_param, target_param in zip(online.parameters(), target.parameters()):
    target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
```

At `tau = 0.005`, the target network moves 0.5% toward the online network each step. This is extremely gentle — the target barely changes at each step, giving training a very stable reference point. This is called **Polyak averaging**.

---

## What you'll notice in the demo

Open the [Rocket Lander ↗](https://huggingface.co/spaces/Dash10107/rocket-lander-sac) — a SAC agent controlling continuous engine thrust to land precisely.

**Three things to watch:**

1. **The action distribution.** Early: the thrust outputs are all over the place — the Gaussian is wide. Late: they converge to smooth, coordinated burns. The standard deviation narrows.

2. **Turbulence recovery.** Enable wind perturbation. Because the SAC policy retains entropy, it doesn't rigidly execute a memorised sequence — it adapts to each perturbation. Compare to a trained DQN agent on the same task: it breaks immediately because it doesn't have the flexibility.

3. **The entropy curve.** Watch `α × H(π)` over training. It starts high (random policy, lots of entropy), then stabilises — not at zero, but at a healthy minimum. The agent stays slightly unpredictable by design.

---

## Try it yourself

**Experiment 1 — Remove entropy (α = 0).**
This turns SAC into a standard Actor-Critic. Training often becomes less stable, and the final policy is more brittle — it works in calm conditions but breaks in turbulence.

**Experiment 2 — High target entropy.**
Force `α` to stay large. The policy stays very random — rewards are lower because the agent explores too much even late in training. But it's remarkably robust to any perturbation.

**Experiment 3 — One critic instead of two.**
Remove the `min` and use a single Q-value estimate. Training initially goes faster (simpler computation) but eventually diverges or plateaus — the actor chases overestimated Q-values into bad regions of the policy space.

---

## Next: what happens when there's more than one agent?

Everything so far has been one agent, one environment. But the most interesting real-world problems have multiple agents — and they affect each other.
