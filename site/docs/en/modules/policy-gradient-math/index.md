---
description: "Policy Gradient Theorem explained from first principles. Covers REINFORCE, the log-probability trick, baseline variance reduction, advantage function, and why all modern RL uses policy gradients."
---

# Policy Gradients Explained — Reinforcement Learning Math
<br> *Why learn the policy directly?*

Everything in Parts 1–3 learned a *value function* — a score for every (state, action) pair. The policy was derived from it: pick the action with the highest Q-value.

This works. But it has a hidden assumption: **you can find the maximum Q-value over all possible actions.**

If you have 4 actions (up, down, left, right), taking the argmax is trivial. If you have 1,000 possible actions, it's slow but doable. If actions are continuous — any real number in a range — the argmax over an infinite set is impossible.

There's also a deeper problem: value-based methods produce deterministic policies. In some problems, the optimal policy is *stochastic* — sometimes you should go left, sometimes right, depending on something the state doesn't capture. A deterministic argmax can never learn that.

Policy gradient methods bypass the value function entirely. They directly parameterise and optimise the policy. This chapter shows you the theorem that makes this possible — and why it's true.

---

## Parameterising the policy

A parameterised policy is a neural network that maps states to action probabilities:

$$\pi_\theta(a | s) = P(\text{take action } a \text{ in state } s \text{; parameters } \theta)$$

The goal is to find the parameters θ that maximise the expected total reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

Where $\tau$ is a trajectory (a full episode: $s_0, a_0, r_1, s_1, a_1, r_2, \ldots$) sampled by running the policy.

We want to compute $\nabla_\theta J(\theta)$ — the direction to adjust the parameters to increase expected reward — and follow it uphill. This is gradient ascent on the expected return.

The question is: how do you compute this gradient? The expected reward involves the environment's transition probabilities, which we don't know.

---

## The Policy Gradient Theorem

Here is the theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right]$$

Where:
- $\nabla_\theta \log \pi_\theta(a_t | s_t)$ = the direction to adjust θ to make action $a_t$ more/less likely in state $s_t$
- $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$ = the actual return received from step $t$ onward
- The expectation $\mathbb{E}$ means we average this over many sampled trajectories

In plain English: *to increase expected reward, adjust the policy parameters in the direction that makes good actions (those followed by high returns) more likely, and bad actions (those followed by low returns) less likely.*

This is the formal version of "do more of what worked." The gradient tells you exactly how much to push each parameter.

---

## Why log probability? The mathematical reason

This might seem arbitrary. Why $\log \pi_\theta$ and not just $\pi_\theta$?

The answer comes from a simple identity:

$$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

This looks like a rearrangement — but it's crucial. When you compute the gradient of the expected reward, the environment's transition probabilities appear in the expectation. They cancel out when you use the log-probability form, leaving an expression that only depends on things we *can* compute: the policy itself and the observed rewards.

You can verify this makes the gradient computable without knowing the environment dynamics — which is exactly what we need for model-free learning.

---

## REINFORCE: the simplest policy gradient algorithm

The theorem immediately gives us an algorithm. Run episodes, collect returns, update the policy:

$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)$$

```python
def reinforce_update(policy, optimizer, episode, gamma=0.99):
    # Compute discounted returns for each timestep
    returns = []
    G = 0
    for reward in reversed(episode.rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # Normalise returns (reduces variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient update
    loss = 0
    for log_prob, G in zip(episode.log_probs, returns):
        loss -= log_prob * G  # negative because we do gradient ascent

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This is REINFORCE (Williams, 1992) — one of the oldest policy gradient algorithms, and still the clearest expression of the theorem.

---

## The variance problem — and why Actor-Critic fixes it

REINFORCE works, but it's slow. The reason: **G_t is very noisy**.

One episode, the agent gets lucky and G_t is high. The next episode, it's unlucky and G_t is low. The policy gradient update swings back and forth with this noise, making training unstable.

The fix is the **baseline**: subtract an estimate b(s) from the return before multiplying.

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t))\right]$$

Crucially, the baseline does not change the expected gradient — it only reduces its variance. This is provable:

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = 0 \quad \text{for any function } b(s)$$

The proof: $b(s)$ doesn't depend on the action, so it factors out of the expectation over actions. And the expected log-gradient of any proper probability distribution sums to zero (because probabilities sum to 1).

The best baseline is $V^\pi(s)$ — the value function. Subtracting it gives the **Advantage**:

$$A(s_t, a_t) = G_t - V^\pi(s_t)$$

In plain English: *not "how good was this action?" but "how much better was this action than what I'd expect on average in this state?"*

This is exactly what the Critic computes in Actor-Critic methods. The Critic estimates $V^\pi(s)$; the Actor uses $(G_t - V^\pi(s_t))$ as the signal. And now you see *why* — it's the variance-reduced form of the policy gradient theorem.

---

## Connecting to every algorithm that follows

Every policy gradient algorithm in this course is a refinement of REINFORCE:

| Algorithm | What they change |
|-----------|-----------------|
| **REINFORCE** | Raw return $G_t$ as the signal |
| **Actor-Critic** | Replace $G_t$ with $r + \gamma V(s')$ (TD target); use $V^\pi$ as baseline |
| **A2C** | Multiple parallel environments; synchronous updates |
| **PPO** | Clip the policy update ratio to prevent large jumps |
| **SAC** | Add entropy bonus; Gaussian policy for continuous actions |

They all share the same gradient direction from the theorem. They differ only in how they estimate $G_t$ (or the advantage), and in what constraints they place on the update step.
