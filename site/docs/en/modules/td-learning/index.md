---
description: "Temporal Difference (TD) learning explained — TD(0), n-step returns, and TD(λ). Learn how TD unifies Monte Carlo and Q-Learning, and how eligibility traces work."
---

# Temporal Difference Learning

Monte Carlo waits until the end of an episode. Q-Learning updates every single step. Both work. But why are they so different? And is there something in between?

Yes. And it turns out the "in between" is where the most powerful ideas live.

---

## The core insight: bootstrapping

Q-Learning updates its estimate using another estimate:

$$\text{Target} = r + \gamma \cdot \max_{a'} Q(s', a')$$

It doesn't wait for the true return (that would be Monte Carlo). It uses its *current best guess* of the next state's value as a stand-in for the future. This is called **bootstrapping** — using your own estimates to improve your estimates.

Bootstrapping is fast. You get a learning signal after every single step. But the quality of the signal depends on how good your current estimates are. If your Q-values are wrong (which they are at the start of training), you're updating toward a noisy target.

Monte Carlo doesn't bootstrap. It waits for the actual return. This makes its targets accurate but slow to compute.

**Temporal Difference (TD) learning** is the family of algorithms that bootstrap — and studying it carefully reveals exactly why Q-Learning works and what its limits are.

---

## TD(0): the simplest version

TD(0) is the simplest TD algorithm. It estimates the value function V^π(s) — not Q(s,a), just V(s) — one step at a time.

After each step, compute the **TD error** δ:

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

Where:
- $r_{t+1}$ = the reward just received
- $\gamma V(s_{t+1})$ = discounted estimate of where you ended up
- $V(s_t)$ = your current estimate of the state you were just in
- $\delta_t$ = the gap between target and estimate

Then update:

$$V(s_t) \leftarrow V(s_t) + \alpha \cdot \delta_t$$

In plain English: *move your estimate a small step toward the target. The step size is α (the learning rate). The target is the reward you got plus your discounted guess about the next state.*

The TD error $\delta_t$ is the most important quantity in all of RL. It's the signal that tells the agent: "your estimate of this state was too high / too low." Every modern RL algorithm — Q-Learning, Actor-Critic, PPO — uses some form of TD error as its core learning signal.

```python
def td0_update(V, state, reward, next_state, alpha, gamma, done):
    # The target: what we wish V(state) had been
    target = reward + gamma * V[next_state] * (not done)

    # The error: how far off were we?
    td_error = target - V[state]

    # Update: move estimate toward target
    V[state] += alpha * td_error

    return td_error  # useful for monitoring training
```

---

## n-step returns: the spectrum

One step is TD(0). An infinite number of steps is Monte Carlo. What about somewhere in between?

The **n-step return** uses the actual rewards for the next *n* steps, then bootstraps from the value estimate at step *n*:

$$G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})$$

Where:
- The first $n$ rewards are real, observed values
- The last term $\gamma^n V(s_{t+n})$ is a bootstrapped estimate from the value function

**At n=1:** this is TD(0). You use one real reward and then bootstrap.
**At n=∞:** this is Monte Carlo. You use all real rewards until the episode ends.
**At any other n:** you're on the spectrum between them.

Larger n means less bias (more real rewards, less reliance on potentially wrong estimates) but more variance (a longer sequence of actual rewards varies more from episode to episode). Smaller n means more bias but less variance.

```python
def n_step_return(rewards, V, next_state, gamma, n):
    """Compute n-step return from a list of rewards."""
    G = 0
    for i, r in enumerate(rewards[:n]):
        G += (gamma**i) * r

    # Bootstrap from the value at step n
    if len(rewards) >= n:
        G += (gamma**n) * V[next_state]

    return G
```

---

## TD(λ): the elegant unification

Instead of picking one value of n, **TD(λ)** combines all n-step returns with exponentially decaying weights:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

Where:
- $\lambda \in [0, 1]$ is the mixing parameter
- $(1-\lambda)$ is a normalising factor that makes the weights sum to 1
- $\lambda^{n-1}$ gives exponentially less weight to longer n-step returns

**At λ=0:** all weight on the 1-step return. This is TD(0).
**At λ=1:** all weight goes to the infinite-step return. This is Monte Carlo.
**At λ=0.9:** 90% of the weight is on longer returns (low bias), 10% on short ones (low variance). A common practical choice.

This one parameter controls the entire bias-variance tradeoff. It's one of the cleanest trade-offs in all of machine learning.

```
λ = 0    TD(0)           High bias, low variance
λ = 0.3  Mostly TD       Medium bias, medium variance
λ = 0.9  Mostly MC       Low bias, high variance
λ = 1    Monte Carlo     Zero bias, high variance
```

---

## Eligibility traces: making TD(λ) efficient

There's a computational problem with TD(λ) as defined above: you can't compute it in real time. You'd need to know all future rewards before you could calculate the weighted average.

The practical solution is **eligibility traces** — a per-state counter that tracks how recently and frequently each state was visited.

When a state is visited, its trace increases. On every step, all traces decay by γλ. The TD error updates all states proportionally to their current trace:

```python
def td_lambda_update(V, states_seen, td_error, alpha, gamma, lmbda):
    """Update all states visited so far, weighted by eligibility traces."""
    for state, trace in states_seen.items():
        V[state] += alpha * td_error * trace
        states_seen[state] *= gamma * lmbda  # decay trace
```

States visited recently (high trace) get large updates. States visited long ago (low trace) get tiny updates. The update naturally propagates backward in time — the same effect as n-step returns, but computed online, one step at a time.

---

## Why this matters

You've now seen the full picture of the bias-variance landscape in RL:

- Q-Learning is TD(0) for Q-values: maximum bootstrap, minimum variance, some bias from wrong estimates.
- Monte Carlo is the zero-bootstrap limit: zero bias from bootstrapping, but high variance from full-episode returns.
- TD(λ) spans the whole spectrum — you can tune to the right spot for your problem.

Modern deep RL algorithms don't always use TD(λ) explicitly. But they all make an implicit choice about where on this spectrum to operate. PPO uses multi-step returns for its advantage estimates. SAC uses 1-step TD targets. Understanding this spectrum means you understand *why* these choices were made.
