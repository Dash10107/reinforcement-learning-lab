---
description: "Learn the multi-armed bandit problem — the explore-exploit tradeoff at the core of all reinforcement learning. Covers epsilon-greedy, epsilon decay, regret minimisation with Python code."
---

# Multi-Armed Bandits Explained — Exploration vs Exploitation
<br> *The problem with guessing.*

You just started a new job. You have five ideas for a project, and you think at least one of them will impress your manager. But you can only present one idea per meeting. And meetings are once a week.

Do you keep presenting your best guess until you're sure it's the winner? Or do you sometimes try one of the other ideas, even if you're less confident, just to find out?

This is the explore-exploit tradeoff. It's one of the oldest problems in decision-making. And it's the first problem every RL agent has to solve — even before it learns anything else.

---

## The Multi-Armed Bandit

The classic version of this problem goes by an odd name: the **multi-armed bandit**.

Picture a row of slot machines. Each machine has a lever (its "arm"). Each machine pays out at a different average rate — but you don't know the rates. You have a limited number of pulls. Which machines do you play?

You could play one machine forever (exploit your current best guess). But maybe you're playing the wrong one. You could try every machine equally (explore everything). But then you waste pulls on bad machines.

The right answer is somewhere in between, and finding it is the whole game.

We use this exact setup — with digital "ads" instead of slot machines — in the project for this chapter.

---

## What does the agent actually do?

At each step, the bandit agent chooses one of several "arms." It gets a reward (drawn from that arm's hidden distribution). It doesn't know the distributions — it can only observe what it gets.

The agent keeps track of how well each arm has done so far:

```python
# Start knowing nothing — equal estimates for all arms
Q = [0.0] * n_arms  # estimated value of each arm
N = [0] * n_arms  # how many times we've pulled each arm


def update(arm, reward):
    N[arm] += 1
    # Running average: blend old estimate with new observation
    Q[arm] += (reward - Q[arm]) / N[arm]
```

The running average is a clean trick: you don't need to store every reward you've ever seen. Just your current estimate and how many times you've sampled. Each new reward nudges the estimate closer to reality.

Formally, the incremental update rule is equivalent to:

$$Q_{n+1}(a) = Q_n(a) + \frac{1}{N_n(a)} \left[R_n - Q_n(a)\right]$$

Where:
- $Q_n(a)$ = estimate of arm $a$'s value after $n$ total steps
- $N_n(a)$ = number of times arm $a$ has been pulled
- $R_n$ = the reward just received
- $\frac{1}{N_n(a)}$ acts as a decaying learning rate — early samples change the estimate a lot; later samples change it very little

In plain English: *the estimate is a running average, and each new observation gets a weight of $1/n$ — the nth pull contributes a nth of the final average.*

---

## Epsilon-greedy: the commonsense solution

The simplest strategy that actually works is called **epsilon-greedy**.

Most of the time (probability `1 - ε`), pick the arm with the highest estimate. This is exploiting. Occasionally (probability `ε`), pick a random arm. This is exploring.

```python
import random


def choose_arm(Q, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(Q) - 1)  # explore: random arm
    else:
        return Q.index(max(Q))  # exploit: best known arm
```

That's it. Two lines that balance the explore-exploit tradeoff.

The key insight: you need *both*. Pure exploitation means you'll never discover if you've missed a better option. Pure exploration means you'll never commit long enough to earn good rewards.

---

## Epsilon decay: learning to commit

Early in training, exploring is cheap. You don't know anything, so random choices are as good as informed ones.

Late in training, exploring is expensive. You've built up good estimates — wasting a pull on a random arm is a real cost.

This is why we **decay epsilon** over time: start high (lots of exploration), reduce it gradually (shift toward exploitation) as the agent builds confidence.

```python
# Epsilon decays from 1.0 down toward 0.01 over training
epsilon = max(0.01, epsilon * 0.995)
```

Watch this in the demo: early episodes look scattered and random. As epsilon falls, the agent commits to its best-performing arm and its average reward climbs.

---

## What you'll notice in the live demo

Open the [Ad Campaign Optimizer ↗](https://huggingface.co/spaces/Dash10107/mab-banner-optimizer) — it runs 5 "ad variants" as bandit arms.

**Three things to watch:**

1. **The reward curve.** Chaotic early, then steadily climbing. That's epsilon decay working.
2. **The arm selection chart.** At first, all arms get roughly equal pulls. By the end, one arm dominates. The agent found the winner.
3. **The regret curve.** Regret is formally defined as:

$$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^{T} r_t$$

Where $\mu^*$ is the true mean reward of the best arm, and $r_t$ is the reward obtained at step $t$. In plain English: *how much total reward did you leave on the table by not always picking the best arm?* Even a simple epsilon-greedy agent grows regret much slower than a purely random strategy.

---

## Try it yourself

**Experiment 1 — Lock in too early.**
Set epsilon to `0.0` from the start. The agent will exploit whatever arm it happened to try first. If it got lucky, great. If not, it's stuck. Watch the regret climb.

**Experiment 2 — Never commit.**
Set epsilon to `1.0` (always explore). The reward never improves because the agent never learns to use what it knows. The arm selection stays flat.

**Experiment 3 — Find the sweet spot.**
Try epsilon values of `0.1`, `0.2`, and `0.3` with the same decay rate. Notice how the right starting value changes depending on how different the arms are. When the best arm is much better than the others, you need less exploration to find it.

---

## What's missing here — and what comes next

The bandit is the simplest possible RL problem. It has no states. The agent's situation doesn't change between pulls. Every pull is independent.

Real environments are not like this. The action you take changes the situation you're in next. Your reward depends not just on what you did, but on *where you were* when you did it.

To handle that, we need memory. We need a way for the agent to remember which situation led to which outcome — not just which arm paid best overall.

That's what Q-Learning gives us. But before we go there — there's one more exploration question worth answering: **can we do better than random guessing?**

Epsilon-greedy is simple. But it doesn't know *why* it's exploring. The next chapter shows two strategies that explore based on *uncertainty* — and consistently outperform epsilon-greedy.
