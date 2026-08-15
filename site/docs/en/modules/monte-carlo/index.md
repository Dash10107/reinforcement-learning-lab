---
description: "Monte Carlo reinforcement learning methods explained. Learn how agents learn from complete episodes, discounted return calculations, and how MC compares to TD learning for bias and variance."
---

# Monte Carlo Reinforcement Learning Tutorial
<br> *Waiting for the ending.*

Q-learning updates after every single step. The moment the agent moves, it updates its estimate of how valuable that move was.

There's something a little suspicious about this. When the agent takes a step in the middle of a maze, it doesn't know yet whether this maze run will end well or badly. It's guessing — updating based on the best Q-value in the next state, which is itself a guess based on the state after that, which is itself a guess.

Q-learning is guesses built on guesses. It works — but it's always working with incomplete information.

Monte Carlo does something completely different. It waits.

---

## The full picture, first

Monte Carlo methods run the entire episode first. The agent wanders through the maze — maybe it finds the exit, maybe it doesn't — and only *after* the episode is completely over does it look back and update.

At that point, it knows the actual return from every state it visited. No guesses. Real numbers.

```
Episode: s₀ → s₁ → s₂ → s₃ → ... → sₙ (exit!)
Rewards:  0    0    0    0         +1

Actual return from s₀: 0 + 0 + 0 + 0 + 1 = 1.0 (discounted)
Actual return from s₁: 0 + 0 + 0 + 1 = 0.99
Actual return from s₂: 0 + 0 + 1 = 0.98
...
```

The agent now has *real* data for every state it visited in this episode. It uses this data to update the Q-values — not estimates, but actual observed returns.

The **discounted return** from step $t$ onward is:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Where:
- $r_{t+k+1}$ = the actual reward received $k$ steps after step $t$
- $\gamma^k$ = the discount applied to rewards further in the future
- The sum runs to the end of the episode (at which point future rewards are 0)

In plain English: *the return at step $t$ is all the rewards from here to the end of the episode, each one worth slightly less than the previous one.*

For the episode above: $G_0 = 0 + 0.99 \times 0 + 0.99^2 \times 0 + 0.99^3 \times 0 + 0.99^4 \times 1 \approx 0.96$

The Monte Carlo update then becomes:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [G_t - Q(s_t, a_t)]$$

In plain English: *move the Q-value estimate toward the real return we observed. The learning rate $\alpha$ controls how fast.*

```python
def update_episode(episode, Q, alpha, gamma):
    G = 0  # discounted return, calculated backwards from the end
    for state, action, reward in reversed(episode):
        G = reward + gamma * G         # actual return from this step onward
        Q[state, action] += alpha * (G - Q[state, action])
```

Notice: we walk backwards through the episode. The last step's return is just its reward. The second-to-last step's return is its reward plus the discounted return from the last step. And so on, all the way back to the beginning.

---

## Why this is better — and worse

**Better:** The returns are real. You're not estimating what comes next — you *know* what came next, because the episode is over. This means the updates are unbiased. No approximation error.

**Worse:** You have to wait for the whole episode. In a long environment — a game that takes hours, a robot task with thousands of steps — waiting for the episode to finish before learning anything is very slow.

And there's another problem: **high variance**. 

One episode of a maze might go well (lucky path, +1 reward). The next might go badly (hit a dead end, 0 reward). Monte Carlo takes both of these seriously and updates its values based on each. If one episode is unusually lucky or unlucky, the updates can be noisy and misleading.

Q-learning has lower variance because it smooths out its estimates step by step. Monte Carlo has lower bias because it uses real returns. This tradeoff — variance vs bias — runs through all of RL.

---

## The bank statement analogy

Think about how you track your spending.

**Q-learning** is like checking your bank balance after every coffee. You always know your current total. But each individual transaction might be misleading — you spent a lot on groceries today, that's not your usual pattern.

**Monte Carlo** is like waiting until the end of the month and reviewing your full statement. You get the complete, accurate picture of the month. But you're always learning about last month.

Neither is strictly better. The right choice depends on your environment. For short episodes with sparse rewards (mazes, small games), Monte Carlo works beautifully. For long, continuous tasks, you want something more like Q-learning.

---

## What you'll notice in the demo

The maze demo lets you switch between Q-learning, SARSA, and Monte Carlo: [Maze Solver ↗](https://huggingface.co/spaces/Dash10107/rl_maze_solver)

**Three things to watch:**

1. **Early episodes.** Monte Carlo needs to *complete* an episode before it learns anything. In a difficult maze, early episodes often fail (no exit found). This means Monte Carlo can have many blank, learning-free episodes at the start. Q-learning learns something from every step, even failed ones.

2. **The Q-value heatmap.** Monte Carlo's heatmap fills in less smoothly than Q-learning's. It updates in chunks — whole episodes at once — rather than continuously. This shows up as "islands" of learned values that slowly spread.

3. **Long-run performance.** After enough episodes, both methods converge to similar policies. The difference is in the path there: Q-learning gets to a decent policy faster; Monte Carlo gets to a *correct* policy more reliably (less bias).

---

## Try it yourself

**Experiment 1 — Short vs long mazes.**
Try Monte Carlo on a 5×5 maze. Then try it on a 15×15 maze. Notice how many episodes it takes to start improving in each. The problem with Monte Carlo in large environments is not intelligence — it's that successful episodes are rare early on, so there's little data to learn from.

**Experiment 2 — Compare variance.**
Run Q-learning and Monte Carlo side by side for 1000 episodes. Plot the episode reward. Monte Carlo's curve is bumpier. That's variance — the full-episode returns bounce around more than the step-by-step estimates.

**Experiment 3 — The lucky first run.**
Reset the environment several times and watch the first 10 episodes of Monte Carlo. Sometimes it gets lucky early and finds the exit. When it does, its Q-values jump significantly. When it doesn't, they stay flat. This volatility is the price of unbiased estimates.

---

## Where this is heading

In practice, most modern RL algorithms are somewhere in between Q-learning and Monte Carlo. The next chapter makes this precise — introducing the family of algorithms that unifies all of them under one framework.
