---
description: "Q-Learning and SARSA explained from scratch. Learn the Bellman optimality equation, on-policy vs off-policy learning, Q-table construction, and convergence conditions with Python code."
---

# Teaching an Agent to Remember

The agent tried going right. It hit a wall.

Tomorrow, going right from that same spot should feel less appealing. Not just less appealing in general — less appealing *from that specific spot*.

This is the key thing bandit agents can't do: they can't connect an action to a place. They just know "arm 3 pays well." They don't know "arm 3 pays well *when you're standing at the crossroads in room 4.*"

To do that, the agent needs a memory. Something that stores, for every situation it might find itself in: "here's how good each action looks from here."

That thing is called a **Q-table**.

---

## The Q-table: a cheat sheet for the whole world

Picture a notebook.

The rows are places — every cell in the maze, every intersection, every room. The columns are moves — up, down, left, right. Each cell in the notebook stores a number: how valuable is this action, from this place?

That number is called a **Q-value**. Q for "quality." It's a score that answers: "if I'm here and I do this, how much total reward can I expect to collect before the episode ends?"

At the start of training, every Q-value is 0. The agent knows nothing. It wanders randomly. But with every step, it learns a little — and it writes what it learned into the notebook. Over time, the notebook becomes a cheat sheet for the whole maze.

```python
import numpy as np

# Q-table: rows = states, columns = actions
# Start with zeros — the agent knows nothing yet
Q = np.zeros((n_states, n_actions))
```

---

## How the agent updates its memory

After each step, the agent runs this calculation:

```
New Q-value = Old Q-value + learning_rate × (Target - Old Q-value)
```

Where the target is:

```
Target = reward I just got + discounted value of my next best move
```

In code:

```python
def update(state, action, reward, next_state, done):
    best_next = np.max(Q[next_state])  # best option from next state
    target = reward + gamma * best_next * (not done)
    Q[state, action] += alpha * (target - Q[state, action])
```

Let's translate this line by line.

`best_next` — the agent looks ahead: "what's the best Q-value in the state I just landed in?" It's making an optimistic assumption: *I'll make the best possible move from here.*

`target` — the reward I just received, plus a discounted guess about future rewards. The discount factor `gamma` (usually 0.99) makes the agent prefer rewards sooner rather than later. A reward now is worth more than a reward ten steps away.

`Q[state, action] += alpha × (target - Q[state, action])` — this is a **running average update**. The term `(target - Q[state, action])` is the error: how wrong was my estimate? We move the estimate a little toward the truth. `alpha` (the learning rate) controls how big that step is.

This formula has a proper mathematical form. In standard notation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

Where:
- $Q(s,a)$ = current estimate of the value of taking action $a$ in state $s$
- $\alpha$ = learning rate (how fast to update; typically 0.1–0.5)
- $r$ = reward just received
- $\gamma$ = discount factor (how much future rewards matter)
- $\max_{a'} Q(s', a')$ = best Q-value in the next state — the "bootstrap" target
- $[\cdots]$ = the **TD error**: how wrong was our current estimate?

This is called the **Bellman optimality equation**, and it's the foundation of Q-learning. But now you know what it's actually saying: *update your estimate based on what just happened, plus your best guess about what comes next.*

---

## Convergence: when does Q-learning actually work?

Q-learning converges to the optimal Q-function — *given enough experience* — under three conditions:

1. **Every state-action pair is visited infinitely often.** The agent must explore enough that every (s,a) pair gets updated many times.
2. **The learning rate decays.** More precisely: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$. In practice, a fixed small α works fine for most problems.
3. **The environment is stationary.** The transition probabilities and rewards don't change over time. (Multi-agent environments break this.)

In practice, condition 1 is handled by epsilon-greedy exploration. Condition 2 is approximated by a fixed, small learning rate. Condition 3 holds for most single-agent environments.

---

## What SARSA does differently

SARSA is Q-learning's more cautious cousin. One thing changes.

Q-learning assumes the agent will make the best possible next move. SARSA assumes the agent will make whatever move it would *actually* make — including random exploration moves.

Formally, SARSA's update is:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma Q(s', a') - Q(s, a)\right]$$

Compare to Q-Learning:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

The only difference: $Q(s', a')$ vs $\max_{a'} Q(s', a')$. SARSA uses the Q-value of the *actually taken* next action; Q-Learning uses the Q-value of the *best possible* next action.

In code:

```python
# Q-Learning: uses the best possible next action (optimistic)
target = reward + gamma * np.max(Q[next_state])

# SARSA: uses the actual next action taken (realistic)
next_action = choose_action(next_state, epsilon)
target = reward + gamma * Q[next_state, next_action]
```

This makes Q-Learning **off-policy** (learning about the greedy policy while following an epsilon-greedy policy) and SARSA **on-policy** (learning about the same epsilon-greedy policy it's following). The effect is real: SARSA learns a safer policy because it accounts for the fact that during training, the agent will sometimes do random things — and so it's more cautious near dangerous spots.

**Rule of thumb**: if your environment has catastrophic failures (robot falling, game ending on a bad move), SARSA. If your environment is forgiving, Q-learning.

---

## Watching the Q-table form

The maze demo is the best way to understand what Q-learning is actually doing. Open it: [Maze Solver ↗](https://huggingface.co/spaces/Dash10107/rl_maze_solver)

**At episode 1:** The Q-table is all zeros. The agent wanders randomly. It probably doesn't find the exit.

**At episode 50:** Small patches of the maze have non-zero Q-values — the exit and the cells immediately around it. The Bellman update has started propagating outward from the goal.

**At episode 500:** The Q-values form a clear gradient. High values near the exit, lower values further away. The agent walks the highest-value path directly to the goal. The heatmap shows a bright trail from start to finish.

This is the Bellman equation in action: reward at the goal → propagates backward → fills the whole maze with information about how close each cell is to the exit.

---

## Try it yourself

**Experiment 1 — Speed up learning.**
Double the learning rate (`alpha = 0.5` instead of `0.1`). The agent learns faster early — but watch what happens in complex mazes. High learning rates cause oscillation. The agent "forgets" what it learned and relearns it repeatedly.

**Experiment 2 — Myopic agent.**
Set `gamma = 0.1` (very low discount). The agent only cares about immediate rewards. In a sparse-reward environment (reward only at the exit), a myopic agent struggles to connect distant actions to the final reward. The Q-values near the exit never propagate far enough.

**Experiment 3 — Q-Learning vs SARSA.**
Switch the algorithm and compare the paths near the walls. SARSA hugs the centre of corridors. Q-learning sometimes cuts closer to walls because it assumes it'll make perfect moves near them. On average, Q-learning wins more. SARSA crashes less.

---

## What Q-learning can't do

Q-learning is powerful, but it has a hard limit: the Q-table.

Every unique state needs a row. A maze with 100 cells has 100 rows. A game with a visual input — where "state" means a 84×84 pixel image — has more unique states than atoms in the observable universe.

You can't build that table.

To handle real-world scale, we need to replace the table with something that can generalise: something that, when it sees a new state it's never visited, can make a reasonable guess about the Q-values based on states it *has* seen.

That's what neural networks do. And that's what brings us to DQN.

But first — one more foundational technique that works very differently from Q-learning.
