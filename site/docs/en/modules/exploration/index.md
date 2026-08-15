---
description: "Learn smarter exploration strategies beyond epsilon-greedy. Covers Upper Confidence Bound (UCB) with full derivation, Thompson Sampling with Beta distributions, and regret comparison."
---

# Smarter Ways to Explore

Epsilon-greedy has a flaw.

When you haven't tried arm 3 much, epsilon-greedy treats it exactly like arm 7 — which you've tried a hundred times and know is mediocre. Both get the same chance of being randomly selected.

That's wasteful. The agent should be curious about *uncertain* options, not random ones.

---

## Optimism under uncertainty

Here's a cleaner philosophy: **be optimistic about things you don't know much about.**

If you haven't tried an action many times, your estimate of its value is uncertain — it could be much better than you think. Give it the benefit of the doubt. Try it. Once you've tried it enough to be confident, stop bothering and exploit your knowledge.

This idea has a beautiful mathematical form: the **Upper Confidence Bound (UCB)**.

---

## UCB: exploration guided by confidence

Instead of picking the arm with the highest estimated value, UCB picks the arm with the highest **optimistic estimate** — the estimated value plus an uncertainty bonus:

$$a_t = \arg\max_a \left[ Q(a) + c \cdot \sqrt{\frac{\ln t}{N(a)}} \right]$$

Where:
- $Q(a)$ = current estimated value of arm $a$
- $t$ = total number of steps taken so far
- $N(a)$ = number of times arm $a$ has been tried
- $c$ = a constant controlling how much we reward exploration (typically 1–2)

The key term is $\sqrt{\ln t / N(a)}$.

**When $N(a)$ is small** (you haven't tried arm $a$ much): the bonus is large. The arm looks very attractive. You'll explore it.

**When $N(a)$ is large** (you've tried arm $a$ many times): the bonus is tiny. You're confident in your estimate. Exploit it or not based on the estimate alone.

**As $t$ grows**: the bonus slowly increases for all arms (because the numerator $\ln t$ grows). This ensures the agent keeps revisiting all options over time — it never permanently ignores an arm. This is the theoretical guarantee UCB provides that epsilon-greedy doesn't.

```python
import numpy as np

def ucb_action(Q, N, t, c=2.0):
    # Avoid division by zero: any arm with N=0 has infinite bonus
    # (we should always try an untried arm first)
    if 0 in N:
        return N.index(0)

    bonuses = c * np.sqrt(np.log(t) / np.array(N))
    optimistic_values = np.array(Q) + bonuses
    return int(np.argmax(optimistic_values))
```

UCB requires no tuning of a decay schedule. The confidence interval naturally tightens as you gather more data. Exploration is automatic.

---

## Thompson Sampling: a probabilistic view

UCB takes a point estimate plus a bonus. Thompson Sampling takes an entirely different approach: **maintain a full probability distribution over what the true value of each arm might be**, and sample from it to choose.

For a binary reward problem (reward is 0 or 1 with some probability), the Beta distribution is a natural model:

$$\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$$

Where:
- $\alpha_a$ = number of successes (reward = 1) on arm $a$ plus 1
- $\beta_a$ = number of failures (reward = 0) on arm $a$ plus 1

At each step, sample one value from each arm's distribution, then pick the arm that sampled highest:

```python
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta  = np.ones(n_arms)   # failures  + 1

    def choose(self):
        # Sample from each arm's Beta distribution
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.alpha[arm] += reward       # success
        self.beta[arm]  += (1 - reward) # failure
```

Why does this work? Initially, all Beta(1,1) distributions are flat — uniform, meaning total uncertainty. Any arm is equally likely to sample high. As you try each arm, the distribution for that arm narrows around its true value. Arms you know are bad have narrow distributions peaked near 0 — they almost never sample high. Arms you've barely tried have wide, flat distributions that sometimes sample high — and when they do, you explore them.

Exploration is *implicit* in the sampling process. You never explicitly decide to explore. You just sample from your beliefs, and uncertainty naturally drives you to try uncertain options.

---

## Comparing the three strategies

| Strategy | Exploration logic | Requires tuning? | Guarantee |
|----------|-------------------|------------------|-----------|
| Epsilon-greedy | Random, uniform probability | Yes — ε and decay | None |
| UCB | Optimistic estimate, decreases with N | Minimal — just c | Sub-linear regret |
| Thompson Sampling | Probabilistic, based on belief | No | Optimal in many cases |

**Regret** is the formal way to measure exploration quality: how much reward did you lose compared to always picking the best arm? UCB and Thompson Sampling both achieve **logarithmic regret** — the loss grows very slowly over time. Epsilon-greedy (without careful tuning) achieves linear regret in the worst case.

---

## Why this matters beyond bandits

Exploration strategies aren't just for bandits. The same question — *should I try something new, or stick with what I know?* — arises at every step of a full RL agent.

In DQN, the same epsilon-greedy logic applies to action selection in the full environment. The same weaknesses apply: random exploration doesn't know which states have been underexplored.

In SAC, the maximum entropy objective is a form of exploration built into the policy — the agent is *always* somewhat uncertain, which drives natural exploration throughout training.

In model-based RL, uncertainty about the world model itself drives exploration — you explore states where the model is least confident, because those are where learning is most valuable.

The bandit exploration problem is a clean, isolated case of a challenge that runs through all of RL. The strategies here — optimism under uncertainty, probabilistic belief — are the core ideas.
