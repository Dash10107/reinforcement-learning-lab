---
description: "Actor-Critic reinforcement learning (A2C) explained. Covers the advantage function, Generalised Advantage Estimation (GAE), combined actor-critic loss with entropy bonus, and Python implementation."
---

# Two Brains Are Better Than One

Everything so far learned a *value* — a score for each action. To choose what to do, the agent looked up the scores and picked the highest one.

That works when you have a small, fixed set of actions. But what if the action space is huge? What if there's no discrete list of actions at all — just a continuous range of numbers?

You can't take the maximum over infinite possibilities.

And there's a deeper problem: value-based methods are *indirect*. They learn values, and hope that good values lead to good decisions. What if we skipped the middleman and learned the decision itself?

That's the idea behind **policy gradient methods** — and Actor-Critic is where it gets practical.

---

## The policy is a network

In Q-learning and DQN, the policy was implicit. You had Q-values; you picked the max. The "policy" was just a rule derived from the values.

Policy gradient methods make the policy explicit: a network that takes a state as input and outputs *probabilities* for each action.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)  # probability for each action
```

The agent samples an action from this distribution, executes it, gets a reward, and then updates the network: *increase the probability of actions that led to good outcomes; decrease the probability of actions that led to bad ones.*

The update rule is: **do more of what worked.**

---

## The problem with pure policy gradients

This sounds clean, but it has a serious issue: **high variance**.

The returns from a policy gradient method are noisy. One good episode might have been lucky. One bad episode might have been unlucky. If you update the policy based on raw returns, it overreacts to noise. Training is slow and unstable.

The solution is to not ask "was this good?" but "was this *better than I expected?*"

That quantity is called the **Advantage**: `A = actual_return - baseline`

If `A > 0`, the action was better than expected. Increase its probability.
If `A < 0`, the action was worse than expected. Decrease its probability.

The question is: what should the baseline be? The best baseline is the **value function** — an estimate of how good the current state is on average.

This is where the second network comes in.

---

## Actor and Critic

**The Actor** is the policy network. It decides what to do.

**The Critic** is the value network. It estimates how good the current state is.

The Actor uses the Critic's estimate to compute the Advantage. The Advantage tells it how much to adjust each action's probability.

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Actor head: outputs action probabilities
        self.actor = nn.Linear(128, n_actions)
        # Critic head: outputs a single value estimate
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return probs, value
```

During training:

```python
# Critic loss: how wrong was the value estimate?
advantage = returns - values.detach()
critic_loss = F.mse_loss(values, returns)

# Actor loss: increase probability of actions with positive advantage
actor_loss = -(log_probs * advantage.detach()).mean()

# Total loss: train both networks together
loss = actor_loss + 0.5 * critic_loss
```

The Critic is the teacher — it gives the Actor feedback in the form of the Advantage. The Actor learns to make decisions that the Critic approves of.

---

## The student-teacher analogy

Here's an analogy that might stick.

The Actor is a student trying to solve problems. The Critic is a teacher watching over their shoulder — not giving the right answer, but saying "that approach was above your usual standard" or "you did worse than I'd expect here."

The student doesn't copy the teacher. They use the teacher's feedback to adjust their strategy. The teacher gets better too — as the student's performance improves, the teacher recalibrates what "average" looks like.

Neither is fixed. Both improve together.

---

## A2C: synchronous, stable, simple

A2C (Advantage Actor-Critic) is the clean, stable version of this idea. The "synchronous" part means: run multiple parallel environments, collect experience from all of them simultaneously, then do one update.

Multiple environments solve the correlation problem that we saw in DQN. Instead of seeing one environment's sequential states, you see many different situations at once. This gives the gradient update more information and less noise.

---

## Generalised Advantage Estimation (GAE)

In practice, A2C doesn't use the raw advantage $A = G_t - V(s_t)$. The raw advantage has high variance because $G_t$ is a full Monte Carlo return.

Instead, it uses **Generalised Advantage Estimation (GAE)**, which smoothly combines TD(0) and Monte Carlo:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the TD error at step $t$.

Where:
- $\delta_t$ = the TD error at step $t$ (how wrong was our value estimate?)
- $\gamma$ = discount factor
- $\lambda$ = the GAE mixing parameter (0 = pure TD, 1 = pure Monte Carlo)
- The sum discounts future TD errors exponentially

In plain English: *instead of using the full episode return (noisy) or just the one-step TD error (biased), blend them: weight recent TD errors heavily and distant ones lightly.*

A common choice is $\lambda = 0.95$ — mostly Monte Carlo, which gives low bias. Combined with $\gamma = 0.99$, this means each TD error gets weighted by $(0.99 \times 0.95)^l \approx 0.94^l$ — decaying to near-zero after ~20 steps.

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalised Advantage Estimation for a trajectory."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        # TD error at step t
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        # Accumulate GAE backward
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

---

## The full A2C objective

A2C minimises a combined loss:

$$\mathcal{L}^{\text{A2C}} = \underbrace{\mathcal{L}^{\text{actor}}}_{\text{policy}} + c_1 \underbrace{\mathcal{L}^{\text{critic}}}_{\text{value}} - c_2 \underbrace{H(\pi_\theta)}_{\text{entropy bonus}}$$

Where:
- $\mathcal{L}^{\text{actor}} = -\mathbb{E}[\log \pi_\theta(a|s) \cdot \hat{A}]$ — push up the probability of high-advantage actions
- $\mathcal{L}^{\text{critic}} = \mathbb{E}[(V_\theta(s) - G_t)^2]$ — make the critic's value estimates accurate
- $H(\pi_\theta) = -\mathbb{E}[\log \pi_\theta(a|s)]$ — entropy: how random is the policy?
- $c_1 \approx 0.5$, $c_2 \approx 0.01$ are weighting coefficients

The entropy bonus prevents the policy from collapsing to a single action too quickly — it encourages the actor to keep exploring even as it gets more confident.

---

## What you'll notice in the demo

Open the [Study Planner ↗](https://huggingface.co/spaces/Dash10107/AI-Tutor-A2C) — an A2C agent scheduling study sessions across subjects.

**Three things to watch:**

1. **The value estimate.** The Critic's output starts near 0 and becomes more confident over time. Watch how it starts to accurately predict which study plans will succeed.

2. **The advantage signal.** Early training: advantages are noisy, large swings. Late training: advantages are smaller and more precise. The Actor is getting better; the Critic has recalibrated what "expected" means.

3. **The policy evolution.** The schedule starts random. Over time, it shifts toward high-priority subjects at peak-focus hours. This is the Actor responding to the Critic's feedback.

---

## Try it yourself

**Experiment 1 — Remove the Critic.**
Set the baseline to 0 (no Critic, just raw returns). Training becomes noisier and slower. The Actor is still learning — but without variance reduction, it takes many more episodes.

**Experiment 2 — Freeze the Actor.**
Train only the Critic for the first 100 episodes, then unfreeze the Actor. The Critic starts with a good baseline, and the Actor can learn from a more stable signal immediately. Often converges faster.

**Experiment 3 — Number of parallel environments.**
Reduce from 8 to 1 parallel environment. Notice how training becomes more volatile — the gradient updates are noisier with fewer simultaneous environments. This is what A3C (the asynchronous version) was designed to fix.

---

## Next: PPO — keeping the updates small

Actor-Critic has one remaining problem: if the policy update is too large, the new policy might be completely different from the old one — and then the Critic's estimates, which were calibrated for the old policy, are wrong.

PPO adds one constraint: **don't let the policy change too much in a single step.** It's a small addition with a large effect.
