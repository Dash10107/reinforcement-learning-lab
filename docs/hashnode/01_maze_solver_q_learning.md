---
title: "Building an AI that Solves Mazes from Scratch (Q-Learning, SARSA & Monte Carlo)"
subtitle: "A deep dive into teaching an AI to escape a maze without a map. We compare Q-Learning, SARSA, and Monte Carlo methods with interactive visual heatmaps."
slug: maze-solver-q-learning
tags: machine-learning, python, artificial-intelligence, data-science
cover: "https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/cover.png"
domain: "reinforcement-learning-dash.hashnode.dev"
---

![Maze AI Cover](https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/cover.png)

When I first started studying Reinforcement Learning (RL), I spent weeks drowning in dense academic papers. I was tired of staring at Greek letters ($\gamma$, $\theta$, $\tau$) and abstract equations without actually *seeing* what they meant. I didn't just want to memorize the Bellman equation; I wanted to touch the knobs, tweak the parameters, and watch the math come to life.

So, I decided to build something highly visual: **a maze solver.**

The problem is classic but surprisingly profound. If you drop an AI agent into a completely dark maze, it has no map. It doesn't know where the walls are, and it certainly doesn't know where the exit is. It only knows its current location. 

To escape, it has to explore, make mistakes, remember what it learned, and gradually build a strategy. In this article, we'll walk through exactly how to build this from scratch using three fundamental algorithms: **Q-Learning, SARSA, and Monte Carlo**. 

If you want to play with the final result while you read, I've hosted the interactive playground here:
👉 **[Live Interactive Demo: RL Maze Solver](https://huggingface.co/spaces/Dash10107/rl_maze_solver)**

---

## 1. Setting the Rules of the World

Before we write any algorithms, we have to define the environment in terms the AI can understand. In RL, this boils down to three things:

*   **State:** The agent's current location. We flatten the grid, so if the agent is in cell (row 2, col 3) on a 5x5 grid, its state is just an integer index.
*   **Action:** The agent can move `Up`, `Down`, `Left`, or `Right`.
*   **Reward:** This is how we communicate our goal to the AI.
    *   Step on open floor: `-1` (We want it to find the exit *fast*).
    *   Bump into a wall: `-5` (Painful. Don't do this).
    *   Reach the goal: `+100` (Success!).

The heavy `-5` wall penalty is the main signal. It forces the algorithms to actually plan clean routes rather than randomly bouncing off the walls to reach the end.

---

### The Dilemma: Exploration vs. Exploitation (The Restaurant Analogy)

Before the AI can learn, it faces the oldest dilemma in Reinforcement Learning: **Exploration vs. Exploitation**. 

Imagine moving to a new city. On your first night, you find a decent pizza place. 
*   If you **Exploit** your knowledge, you will eat decent pizza every single night. You will never starve, but you will also never experience anything better.
*   If you **Explore**, you might suffer through a terrible salad, but you might also discover a 5-star steakhouse. 

To solve this, our AI uses an **Epsilon-Greedy** strategy. It is the mathematical engine of curiosity. 

```python
import numpy as np

def choose_action(state, Q_table, epsilon):
    # EXPLORE: Roll a loaded die
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(4) # Pick a completely random direction
        
    # EXPLOIT: Use the cheat sheet
    else:
        return np.argmax(Q_table[state]) # Pick the mathematically best direction
```

When training starts, `epsilon` is `1.0` (100% exploration). The AI stumbles blindly into walls. As it learns, `epsilon` decays to `0.01`. It becomes a master of the maze, only exploring 1% of the time just in case there is a slightly faster shortcut it missed.

---

## 2. Q-Learning: The Optimistic Explorer

Q-Learning is arguably the most famous tabular RL algorithm. The "tabular" part means the agent literally maintains a massive cheat sheet—a table with one row for every cell in the maze and one column for every possible action. 

Each entry in this table is a **Q-value**: a numerical score of how good the agent currently believes it is to take a specific action from a specific cell.

The magic of Q-Learning lies in its update rule. After every single step, the agent updates its cheat sheet using the **Bellman Equation**:

**Q(s, a) ← Q(s, a) + α * [ R + γ * max Q(s', a') - Q(s, a) ]**

In plain English, the math says: *The value of my current move is the immediate reward I just got, plus the discounted value of the absolute best possible move from wherever I ended up next.*

Here is the entire training loop in Python. Notice how short the core logic is:

```python
def train_qlearning(env, episodes, alpha, gamma, decay):
    agent = TabularAgent(env.n_states, env.action_space.n, alpha, gamma)
    
    for _ in range(episodes):
        state, _ = env.reset()
        
        # Walk through the maze until we find the exit or time out
        for _ in range(env.n_states * 4):
            action = agent.choose_action(state) # Epsilon-greedy
            next_state, reward, done, _, _ = env.step(action)
            
            # The core Q-Learning update rule
            # Notice the np.max() - it bootstraps from the *best* possible next move
            td_target = reward + gamma * np.max(agent.Q[next_state]) * (1 - done)
            agent.Q[state, action] += alpha * (td_target - agent.Q[state, action])
            
            state = next_state
            if done: break
                
        agent.decay_epsilon(decay)
```

Because Q-Learning uses `np.max(agent.Q[next_state])`, it is highly optimistic. It updates its knowledge assuming it will always take the perfect path in the future. This is known as being **off-policy**.

---

## 3. SARSA: The Cautious Learner

But what if our agent's current strategy involves a lot of random exploration (which it does, thanks to our epsilon-greedy logic)? 

If Q-Learning is exploring randomly, it might stumble into a wall and take a `-5` penalty. But when it updates its Q-table, it shrugs it off, pretending it would have taken the optimal path instead. 

**SARSA (State-Action-Reward-State-Action)** takes a different approach. Instead of updating toward the *maximum* possible future value, it updates toward the value of the action it *actually* chose next.

Here is the mathematical update rule for SARSA:

**Q(s, a) ← Q(s, a) + α * [ R + γ * Q(s', a') - Q(s, a) ]**

Notice the subtle difference from Q-Learning? There is no `max` operator. The code difference is literally one line, but the behavioral difference is massive:

```python
# SARSA's on-policy TD update
# We don't assume the best possible next move. 
# We evaluate the action we are ACTUALLY going to take next (a').
td_target = reward + gamma * agent.Q[next_state, next_action] * (1 - done)
```

Because SARSA learns the value of the policy it is currently running (random mistakes included), it is much more cautious. If an agent randomly stumbles into walls while navigating a tight corridor, SARSA will learn to lower the value of that corridor and might prefer a wider, safer path. Q-learning will confidently march right next to the wall, assuming it won't make a mistake.

---

## 4. First-Visit Monte Carlo: Hindsight is 20/20

Both Q-Learning and SARSA update their cheat sheets after every single step. **Monte Carlo** methods take a completely different philosophical approach: *Wait until the run is over.*

Instead of bootstrapping values step-by-step, a Monte Carlo agent plays out a full episode to the very end, storing every `(state, action, reward)` in memory. Then, it works backward, calculating the actual cumulative return `G` and updating the values based on real outcomes:

```python
G = 0
# Work backwards from the end of the episode to the start
for state, action, reward in reversed(episode_memory):
    G = reward + (gamma * G)
    
    # Update the Q-Table toward the actual experienced return (G)
    agent.Q[state, action] += alpha * (G - agent.Q[state, action])
```

*   **The Pro:** It is completely unbiased. It uses real experienced returns rather than mathematical estimates.
*   **The Con:** It has incredibly high variance. A single unlucky episode where the agent gets lost can temporarily ruin the values of perfectly good states. Furthermore, because it cannot update until the episode ends, it struggles massively in large mazes where initial episodes might take thousands of steps.

---

## 5. Visualizing the "Brain"

![Q-Learning Heatmap Progress](https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/maze_preview.png)

When building this project, the most satisfying part wasn't writing the code; it was rendering the **State-Value Heatmaps**.

If you look at the Q-table directly, it's just a giant matrix of floating-point numbers. But if you take the maximum Q-value for each cell and map it to a color gradient, you can literally watch the AI's brain light up. 

During training, the cells right next to the goal turn bright yellow first. Then, slowly, episode by episode, that bright path creeps backward through the maze corridors toward the start line. You are watching the reward signal propagate through the environment. It's a beautiful transition from pure random fumbling to precise mathematical pathfinding.

---

## 🧪 Try It Yourself

The best way to understand these trade-offs is to see them in action. I built an interactive dashboard where you can race these algorithms against each other. 

Open up the **[Live Playground](https://huggingface.co/spaces/Dash10107/rl_maze_solver)** and try these three experiments:

1. **The Cautious Agent Test:** Run both SARSA and Q-Learning on `Medium (9x9)` difficulty for 400 episodes. Look at the heatmaps for the cells directly adjacent to the walls. You will see Q-Learning assigns them much higher values than SARSA does!
2. **The Size Limit:** Run Q-Learning for 500 episodes on `Tiny (5x5)` and then on `Large (13x13)`. You'll quickly see why exploration decay parameters are so critical as the state space grows.
3. **The Algorithm Race:** Go to the "Algorithm Race" tab, select a `Large (13x13)` DFS maze, and run all three algorithms simultaneously. Watch how Monte Carlo's convergence curve lags behind the TD methods because it has to wait for those massive initial episodes to finish.

---

### Final Thoughts

The maze solver is a perfect microcosm of Reinforcement Learning. It's small enough to solve on your laptop in seconds, but complex enough to visually demonstrate the core tension in RL: exploration vs. exploitation, and on-policy vs. off-policy learning.

This is just the first of 12 interactive RL projects I'm building to demystify artificial intelligence. If you found this breakdown helpful, the highest compliment you can give is checking out the source code and dropping a star on the GitHub repository:

⭐ **[Reinforcement Learning Lab on GitHub](https://github.com/Dash10107/reinforcement-learning-lab)**

Let me know in the comments: *Have you ever tried to implement Q-learning? What was the hardest part to wrap your head around?*
