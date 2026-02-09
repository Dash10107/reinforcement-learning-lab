---
title: Rl Maze Solver
emoji: 🤖
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# RL Maze Solver — Interactive Reinforcement Learning Playground

An interactive playground where you watch three different reinforcement learning algorithms learn to escape mazes in real time. Choose a difficulty, pick an algorithm, and watch the animated replay showing each step the agent took to find the exit. An algorithm race mode lets all three strategies compete on the same maze so you can see their strengths and weaknesses directly.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/rl_maze_solver)

---

## What problem does this solve?

A maze is a grid of cells and walls. An agent starts at the top-left corner and needs to reach the bottom-right corner. It cannot see the full maze — it only knows where it is right now. Every wall it bumps into is a penalty. Every step it takes costs time. Reaching the goal gives a large reward.

The agent has no map. It has to explore, remember what it learned, and gradually build up a strategy for reaching the goal efficiently. This is the core loop of reinforcement learning: act, observe the outcome, update your knowledge, repeat.

The maze problem is small enough to solve quickly and simple enough to understand visually, which makes it the ideal environment for comparing how different RL algorithms approach the same task.

---

## The Three Algorithms

### Q-Learning

Q-Learning is the most famous tabular RL algorithm. It maintains a table with one row per state (maze cell) and one column per action (up, down, left, right). Each entry Q(s, a) represents how good the algorithm currently believes it is to take action a from state s.

The update rule after each step:

```
Q(s, a) ← Q(s, a) + alpha * (reward + gamma * max Q(s', all_actions) - Q(s, a))
```

This is called the Bellman equation. It says the value of (state, action) should equal the immediate reward plus the best value achievable from the next state, discounted by gamma. The key word is max — Q-learning always bootstraps from the best possible next action, even if the agent did not actually take that action. This makes Q-learning off-policy.

The agent explores using epsilon-greedy: with probability epsilon it picks a random action, otherwise it picks the action with the highest Q-value. Epsilon decays over time so the agent explores a lot early and exploits what it has learned later.

### SARSA

SARSA is almost identical to Q-Learning, but with one important difference. Instead of updating toward the maximum Q-value of the next state, it updates toward the Q-value of the action it actually chose next:

```
Q(s, a) ← Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
```

where a' is the actual next action selected by the epsilon-greedy policy. This makes SARSA on-policy — it learns the value of the policy it is currently running, including its random exploration moves.

In practice this makes SARSA more cautious near dangerous areas. If the agent tends to randomly stumble into high-penalty walls near a certain path, SARSA will learn to avoid that path. Q-learning, which pretends it would always take the greedy action, may not avoid it. In the maze environment SARSA's wall-bump penalty of -5 makes this distinction visible.

### First-Visit Monte Carlo

Monte Carlo methods take a completely different approach. Instead of updating after every step, the agent plays out a full episode to the end (or times out), then uses the actual cumulative reward it received to update every state-action pair it visited.

The return G for a visit to (s, a) at time t is:

```
G_t = reward_t + gamma * reward_(t+1) + gamma^2 * reward_(t+2) + ...
```

The Q-value is updated as a running average of all returns seen from that state-action pair. First-visit Monte Carlo only counts the first time (s, a) appears in each episode to avoid double-counting.

Monte Carlo is unbiased — it uses real experienced returns rather than bootstrapped estimates. But it is high variance because a single unlucky episode can temporarily give a state a very bad update. It also cannot update until the episode ends, which is why it can be slow to learn in long mazes. However, once it does converge, its estimates are accurate.

---

## The Maze Environment

Two maze generation styles are available.

**DFS Corridors:** Uses a depth-first search backtracker to carve out a proper maze with winding corridors. Every cell is reachable from every other cell — the maze is guaranteed solvable. This produces mazes that look like what you would draw by hand, with clear paths and dead ends.

**Open Field (random walls):** Places walls randomly at about 18% of cells. The result is more open with multiple viable routes. Less challenging structurally but good for seeing how agents handle multiple equivalent paths.

**State:** Each cell in the grid is assigned a unique integer index (row × width + column). The agent's state is just this index — a single number.

**Reward per step:**

| Event | Reward |
|---|---|
| Step on open floor | -1 |
| Bump into a wall | -5 |
| Reach the goal | +100 |

The -5 wall penalty is the main signal that separates the algorithms. It teaches the agent to plan routes that avoid walls rather than randomly bouncing around.

---

## Difficulty Presets

| Preset | Grid size | Challenge |
|---|---|---|
| Tiny (5×5) | 5×5 | 25 cells — algorithms converge in under a second |
| Small (7×7) | 7×7 | 49 cells — visible difference between algorithms |
| Medium (9×9) | 9×9 | 81 cells — good for algorithm race comparison |
| Large (13×13) | 13×13 | 169 cells — Monte Carlo struggles, Q-learning shines |
| XL (17×17) | 17×17 | 289 cells — requires more training episodes |

---

## Project Structure

```
rl_maze_solver/
├── app.py                  main Gradio application
├── maze/
│   ├── generator.py        DFS maze generation and open-field generation
│   └── env.py              MazeEnv — Gymnasium-compatible environment
├── agents/
│   ├── base.py             TabularAgent base class with Q-table and epsilon-greedy
│   ├── qlearning.py        Q-Learning training loop
│   ├── sarsa.py            SARSA training loop
│   └── montecarlo.py       First-Visit Monte Carlo training loop
├── viz/
│   └── renderer.py         maze frame rendering, GIF export, training charts
└── requirements.txt
```

---

## Quick Setup

```bash
git clone https://github.com/yourusername/rl-portfolio
cd rl_maze_solver
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

**Suggested starting point:** Go to the Playground tab, select Medium (9x9) difficulty, choose Q-Learning with 800 episodes, and click Train and Watch. An animated GIF will show the agent walking the maze after training. Then go to the Algorithm Race tab, set the same difficulty, and click Start Race. All three algorithms train on identical mazes and their reward convergence curves are shown side by side.

---

## What each tab shows

**Playground:** Pick difficulty, maze style, algorithm, and training episodes. Click Train and Watch to run. The animated GIF shows the agent's final learned path through the maze, with colour-coded cells showing route, start, and goal. Below the GIF, a training progress chart shows reward per episode and a Q-value heatmap shows what the agent learned about each cell.

**Algorithm Race:** Run Q-Learning, SARSA, and optionally Monte Carlo on the same maze. A convergence chart shows all three reward curves simultaneously. A bar chart compares final performance. This is the most informative view for understanding the differences between algorithms.

**How It Works:** Explains all three algorithms, the maze generation choices, the reward structure, and how to interpret each chart.

---

## Reading the Q-Value Heatmap

After training, the Q-value heatmap shows the maximum Q-value the agent assigns to each cell (the value of the best action from that cell). Brighter cells are ones the agent has learned to prefer — they lie on or near the optimal path. Dark cells near walls show where the agent has learned low values because reaching those cells typically leads to wall bumps or dead ends.

A well-trained agent will show a clear bright corridor from start to goal.

---

## Requirements

```
gradio>=6.0.0
numpy
matplotlib
```

---

## Things to Try

**1. Watch convergence speed change with maze size.**
Run Q-Learning for 500 episodes on Tiny (5x5), Medium (9x9), and Large (13x13). The large maze barely improves in 500 episodes. How many episodes does it take Large to reach the performance Tiny achieves in 100?

**2. Compare SARSA vs Q-Learning near walls.**
Run both on Medium for 400 episodes. Look at the Q-value heatmaps for cells adjacent to walls. Q-Learning assigns them higher values than SARSA because it bootstraps from the greedy action rather than the actual cautious one.

**3. Run the Algorithm Race with Monte Carlo included.**
On Large (13x13) DFS, run all three with 800 episodes. Monte Carlo converges slowest because it cannot update until an episode ends — and long mazes have long episodes. Notice its final accuracy may still be good even if slow.

**4. Compare DFS Corridors vs Open Field.**
Run Q-Learning on both maze styles for 500 episodes. In the open field you might see multiple bright corridors in the Q-value heatmap — several equivalent routes the agent has learned to value.

**5. Run for just 50 episodes then watch the animation.**
The agent will often fail to reach the goal or take an obviously suboptimal path. Then run 800 episodes and compare. This before-and-after is the clearest illustration of what training actually does.

---

## Further Reading

- Watkins and Dayan, Q-Learning (1992) — the paper that introduced Q-Learning
- Sutton and Barto, Reinforcement Learning: An Introduction — Chapter 5 for Monte Carlo, Chapter 6 for TD and SARSA/Q-Learning
- Rummery and Niranjan, On-Line Q-Learning Using Connectionist Systems (1994) — introduced SARSA under the name modified connectionist Q-learning
