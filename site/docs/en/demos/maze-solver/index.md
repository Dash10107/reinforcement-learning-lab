---
description: "Maze Solver demo — watch Q-Learning, SARSA, and Monte Carlo race through procedurally generated mazes in real time. See the Q-value heatmap fill as the agent learns."
---

# 🧩 Maze Solver

Watch three algorithms — Q-Learning, SARSA, and Monte Carlo — race through a procedurally generated maze in real time. Switch between them and see what makes each one different.

<a href="https://huggingface.co/spaces/Dash10107/rl_maze_solver" target="_blank" rel="noopener" class="hero-btn-primary" style="display:inline-flex;margin-bottom:1.5rem">
  ▶ Open Live Demo ↗
</a>

---

## What this demo shows

**The Q-value heatmap.** The grid is colour-coded: each cell's brightness represents how valuable the algorithm thinks that cell is. Watch the heatmap fill in from the exit outward — that's the Bellman equation propagating reward backward through the maze.

**The policy arrows.** Switch to "Policy view." Each cell shows an arrow indicating the greedy action — the best move the agent knows. Early training: arrows point randomly. Late training: all arrows form a continuous path to the exit.

**Three algorithms, one maze.** The key comparison isn't raw performance — it's *how* each algorithm learns:

| Algorithm | Updates when? | Follows which policy? | Safer near walls? |
|-----------|--------------|----------------------|-------------------|
| Q-Learning | Every step | Greedy (optimal) | No — optimistic |
| SARSA | Every step | Behavioural (ε-greedy) | Yes — conservative |
| Monte Carlo | End of episode | — (uses full returns) | No — high variance early |

---

## Try these experiments

**Experiment 1 — Short vs long maze.**
Set maze size to 5×5. Watch all three converge quickly. Now switch to 15×15. Monte Carlo's learning curve flatlines early — it needs successful episodes to learn, and those are rare at first in a large maze.

**Experiment 2 — Cliff edge.**
Enable the "cliff" mode (penalty −10 for falling). Q-Learning will still walk near the edge because it assumes optimal future play. SARSA will stay further back — it knows the agent will sometimes explore randomly, so it accounts for the risk.

**Experiment 3 — Compare variance.**
Run Q-Learning and Monte Carlo for 500 episodes each with the same random seed. Plot their episode reward curves. Monte Carlo's curve is bumpier. That's variance: full-episode returns bounce around more than step-by-step TD estimates.

---

## The chapter behind this demo

This demo is paired with two chapters:

- **[Teaching an Agent to Remember](../../modules/q-learning/)** — covers Q-Learning and SARSA in full, including the Bellman equation and on-policy vs off-policy learning
- **[Waiting for the Ending](../../modules/monte-carlo/)** — covers Monte Carlo methods, the discounted return formula, and the bias-variance tradeoff

---

**Difficulty:** Beginner · **Algorithms:** Q-Learning, SARSA, Monte Carlo · **Environment:** Grid Maze
