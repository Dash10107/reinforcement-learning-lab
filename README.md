# Reinforcement Learning Portfolio

A collection of 12 end-to-end reinforcement learning projects, each deployed as an interactive web application on Hugging Face Spaces. The projects span the full range of modern RL — from the simplest tabular methods that fit on a single page, to multi-agent coordination, model-based planning, and learning from human feedback.

Every project is built to be understood by someone who is new to RL. Each has its own README explaining the algorithm, the environment, and what you are looking at when you run it.

**New to reinforcement learning?** Start with these two documents before anything else:

- [CONCEPTS.md](./CONCEPTS.md) — what RL is, the core vocabulary, and how all 12 algorithms relate to each other
- [GETTING_STARTED.md](./GETTING_STARTED.md) — step-by-step guide to running your first project and your first experiment

---

## What is in this repository

| Project | Algorithm | What it does |
|---|---|---|
| [AI-Tutor-A2C](#ai-tutor-a2c) | Actor-Critic (A2C) | Recommends which subject a student should study next |
| [Digital Calligrapher RLHF](#digital-calligrapher-rlhf) | RLHF / Bradley-Terry | Learns your aesthetic preferences for brush strokes |
| [Green Logistics Optimizer](#green-logistics-optimizer) | Deep Q-Network (DQN) | Routes delivery vehicles through a city to minimise emissions |
| [MAB Banner Optimizer](#mab-banner-optimizer) | 6 Bandit Algorithms | Finds the best-performing ad banner with minimal wasted impressions |
| [Market Regime Detector](#market-regime-detector-hmm) | Gaussian HMM | Identifies Bull, Bear, Neutral, and Crisis phases in stock data |
| [MARL Warehouse Sim](#marl-warehouse-sim) | Independent PPO (IPPO) | Trains a team of warehouse robots to coordinate package deliveries |
| [MBRL Pendulum Playground](#mbrl-pendulum-playground) | Model-Based RL + MPC | Learns a world model of a pendulum and plans inside it |
| [RL Maze Solver](#rl-maze-solver) | Q-Learning, SARSA, Monte Carlo | Three tabular RL algorithms race to solve generated mazes |
| [Rocket Lander SAC](#rocket-lander-sac) | Soft Actor-Critic (SAC) | Lands a rocket using continuous engine throttle control |
| [Smart Grid Energy Optimizer](#smart-grid-energy-optimizer) | DQN + Dynamic Programming | Manages a battery to buy electricity cheap and sell it at peak |
| [Swarm Architect MARL](#swarm-architect-marl) | Independent PPO (IPPO) | Five agents cooperatively spread across a 2D space to cover landmarks |
| [Unity RL Huggy Demo](#unity-rl-huggy-demo) | PPO (Unity ML-Agents) | A 3D dog character trained to fetch a stick using physics simulation |

---

## What you get from this repository

**A full progression from simple to advanced RL.** The projects are not isolated demos — they form a learning path. Start with the Maze Solver to understand Q-learning and tabular methods, move to the Rocket Lander for continuous control, and work up to multi-agent coordination and model-based planning.

**Every algorithm implemented cleanly.** Each project has a modular structure separating the environment, the agent, the visualisation, and the application layer. The code is readable and well-commented — not just a research script but something you can study and modify.

**Interactive web apps for every project.** All 11 projects are deployed on Hugging Face Spaces. You can run them in the browser without installing anything. Each has training capabilities built in — you can retrain agents from scratch, adjust hyperparameters, and see results in real time.

**Beginner-friendly documentation.** Every project README explains the algorithm in plain English before showing any equations. The goal is that someone who has heard of reinforcement learning but never implemented it should be able to read any project README and understand what is happening.

---

## Project Summaries

### AI Tutor A2C

An A2C agent that learns to recommend which subject a student should study next across five subjects (Mathematics, Physics, Literature, History, Computer Science). The environment models both learning gains and forgetting — ignoring a subject for too long causes it to decay. The agent must find a study schedule that keeps all subjects progressing toward mastery.

**Algorithm:** Actor-Critic (A2C) — the actor outputs a probability over subjects, the critic estimates state value, and the advantage function tells the actor which choices were better than expected.

[View project](./AI-Tutor-A2C) · [Live demo](https://huggingface.co/spaces/Dash10107/AI-Tutor-A2C)

---

### Digital Calligrapher RLHF

A demonstration of Reinforcement Learning from Human Feedback. Two calligraphic brush strokes are shown side by side and you vote for the one that feels more elegant. A Bradley-Terry reward model updates after each vote, learning a 6-dimensional weight vector that represents your aesthetic preferences. After enough votes, a hill-climbing optimiser generates strokes that maximise your learned reward.

**Algorithm:** RLHF with Bradley-Terry pairwise comparison model — the same technique used to align large language models with human values, applied to visual aesthetics.

[View project](./digital-calligrapher-rlhf) · [Live demo](https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf)

---

### Green Logistics Optimizer

A DQN agent navigates a city grid to make deliveries while minimising carbon emissions. High-congestion zones multiply fuel cost by 4x. The agent learns to route around them when the detour is worth it. Three strategies run on the same city: the DQN agent, a greedy heuristic that always moves toward the goal, and an A* shortest-path solver.

**Algorithm:** Deep Q-Network (DQN) with experience replay and target network — the agent learns a Q-function mapping (position, congestion layout) to expected future reward for each direction.

[View project](./green-logistics-optimizer) · [Live demo](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer)

---

### MAB Banner Optimizer

Six multi-armed bandit algorithms compete on the same ad campaign: Epsilon-Greedy, Decaying Epsilon-Greedy, UCB1, Thompson Sampling, Gradient Bandit, and EXP3. Each algorithm sees the same banner impressions and tries to figure out which banner variant has the highest click-through rate. Their cumulative reward and regret curves are plotted side by side. A step-through learner mode lets you watch each algorithm make individual decisions.

**Algorithms:** All six major bandit strategies — from the simplest (epsilon-greedy) to the Bayesian (Thompson Sampling) and the adversarial (EXP3).

[View project](./mab-banner-optimizer) · [Live demo](https://huggingface.co/spaces/Dash10107/mab-banner-optimizer)

---

### Market Regime Detector HMM

A Gaussian Hidden Markov Model identifies recurring market states in historical stock data: Bull (positive drift, low volatility), Neutral (mixed signals), Bear (negative drift), and Crisis (extreme volatility). The price chart is coloured by regime, the transition matrix shows how likely each regime is to follow each other, and a backtest compares a regime-switching trading strategy against buy-and-hold.

**Algorithm:** Gaussian HMM trained via Baum-Welch EM, decoded via Viterbi. BIC model selection picks the optimal number of hidden states automatically.

[View project](./market-regime-detector-hmm) · [Live demo](https://huggingface.co/spaces/Dash10107/market-regime-detector-hmm)

---

### MARL Warehouse Sim

A team of robots operates inside a custom 12x12 warehouse grid. Each robot picks up packages from loading zones and delivers them to drop-off zones. Robots trained with Independent PPO learn to spread across the warehouse and claim different tasks without colliding, rather than all clustering around the same pickup zone.

**Algorithm:** Independent PPO (IPPO) — each robot maintains its own Actor-Critic pair with no shared parameters or centralised communication.

[View project](./marl-warehouse-sim) · [Live demo](https://huggingface.co/spaces/Dash10107/marl-warehouse-sim)

---

### MBRL Pendulum Playground

A complete Model-Based RL system for the classic Pendulum control task. A neural ensemble of five dynamics models is trained on random exploration data, then used for planning via Random Shooting MPC: at each step, 512 random action sequences are rolled out in the learned model and the best one is selected. An imagination rollout visualiser shows how prediction error grows with horizon length — the central challenge of model-based planning.

**Algorithm:** Ensemble dynamics model (bootstrap training, epistemic uncertainty via model disagreement) + Random Shooting MPC with discounted reward planning.

[View project](./mbrl-pendulum-playground) · [Live demo](https://huggingface.co/spaces/Dash10107/mbrl-pendulum-playground)

---

### RL Maze Solver

Three tabular RL algorithms are trained to navigate procedurally generated mazes: Q-Learning (off-policy TD), SARSA (on-policy TD), and First-Visit Monte Carlo. The playground tab shows the animated solution path and Q-value heatmap for any chosen algorithm. The algorithm race tab plots their convergence curves side by side on the same maze.

**Algorithms:** Q-Learning, SARSA, and Monte Carlo — the three foundational methods from Sutton and Barto's RL textbook, all implemented from scratch using only NumPy.

[View project](./rl_maze_solver) · [Live demo](https://huggingface.co/spaces/Dash10107/rl_maze_solver)

---

### Rocket Lander SAC

A Soft Actor-Critic agent controls two engine throttles to land a rocket on the LunarLander-v3 platform. SAC maximises both expected reward and policy entropy, which leads to natural, exploratory behaviour during training and robust performance at test time. The app shows animated episode replays with throttle overlays, flight trajectories, and full mission analytics. A fine-tuning lab lets you continue training the pre-trained model in the browser.

**Algorithm:** SAC with dual critics, target networks, and automatic entropy tuning — one of the best off-policy algorithms for continuous control tasks.

[View project](./rocket-lander-sac) · [Live demo](https://huggingface.co/spaces/Dash10107/rocket-lander-sac)

---

### Smart Grid Energy Optimizer

A DQN agent manages a battery energy storage system over a 24-hour electricity market cycle. It decides each hour whether to charge (buy electricity), discharge (sell or offset load), or hold. Solar generation and building load vary throughout the day. The agent is compared against a dynamic programming optimal solver (perfect foresight upper bound) and a price-threshold heuristic baseline.

**Algorithm:** DQN for the learned policy; DP backward induction for the optimal solver; price-threshold rule for the heuristic baseline.

[View project](./smart-grid-energy-optimizer) · [Live demo](https://huggingface.co/spaces/Dash10107/smart-grid-energy-optimizer)

---

### Swarm Architect MARL

Five agents learn to cooperatively spread across a continuous 2D space to cover five landmarks. No agent is told which landmark to target — they must implicitly divide the landmarks among themselves through experience. The task comes from PettingZoo's Multi-Particle Environment suite and is a standard benchmark for cooperative multi-agent RL.

**Algorithm:** Independent PPO (IPPO) — same algorithm as the Warehouse Sim but in a continuous action space with a different coordination challenge.

[View project](./swarm-architect-marl) · [Live demo](https://huggingface.co/spaces/Dash10107/swarm-architect-marl)

---

### Unity RL Huggy Demo

A pre-trained Unity ML-Agents character that learned to fetch a stick through millions of physics simulation steps. Huggy controls joint torques on a physically simulated dog body and must discover running, turning, and timing entirely from a sparse reward signal. The demo runs as a WebGL application in the browser.

**Algorithm:** PPO via Unity ML-Agents — the same policy gradient algorithm used in several other projects, applied to a 3D continuous control problem.

[View project](./Unity-RL-Huggy-Demo) · [Live demo](https://huggingface.co/spaces/Dash10107/Unity-RL-Huggy-Demo)

---

## Getting Started

### Run any project individually

Each project has its own `requirements.txt` and can be run standalone:

```bash
cd rocket-lander-sac
pip install -r requirements.txt
python app.py
```

### Set up a shared environment for all projects

A root-level `requirements.txt` and setup script installs everything needed for all 11 projects into a single virtual environment:

**On Windows (PowerShell):**

```powershell
.\setup_env.ps1
```

**On Mac or Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then activate the environment before running any project:

```powershell
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

---

## Suggested Learning Order

If you are new to reinforcement learning, this order builds understanding progressively:

1. **RL Maze Solver** — Start here. Q-learning is the simplest RL algorithm. Watching the maze solve itself makes the core loop (act, observe, update) immediately visual and concrete.

2. **MAB Banner Optimizer** — Multi-armed bandits are simpler than full RL (no state transitions). Understanding exploration vs exploitation here makes the rest of the portfolio easier to follow.

3. **AI Tutor A2C** — Your first policy gradient algorithm. A2C adds the actor-critic architecture on top of the tabular intuition from the maze solver.

4. **Rocket Lander SAC** — Continuous actions for the first time. SAC is more complex than A2C but the visual feedback of a landing rocket makes the learning progress tangible.

5. **Green Logistics Optimizer** — DQN applied to a spatial planning problem. Good for understanding experience replay and how Q-networks generalise across states.

6. **Smart Grid Energy Optimizer** — DQN in a real-world economic setting. Introduces the comparison between learned policy and optimal (DP) policy.

7. **Swarm Architect MARL** — First multi-agent project. Independent learning on a cooperative task.

8. **MARL Warehouse Sim** — Multi-agent with a richer task structure. Task assignment, collision avoidance, and throughput metrics.

9. **MBRL Pendulum Playground** — The shift to model-based RL. Understanding world models, planning, and compounding prediction errors.

10. **Market Regime Detector** — Statistical ML applied to RL's state estimation problem. Good bridge between classical ML and RL.

11. **Digital Calligrapher RLHF** — RLHF connects RL to how modern AI systems are aligned with human preferences. A fitting endpoint.

---

## Tech Stack

| Category | Libraries |
|---|---|
| RL algorithms | Stable-Baselines3, custom PyTorch implementations |
| Environments | Gymnasium, PettingZoo, custom environments |
| Statistical ML | hmmlearn, scikit-learn |
| Financial data | yfinance |
| Visualisation | matplotlib, PIL |
| Web interface | Gradio |
| Deep learning | PyTorch |

---

## Repository Structure

```
ReinforcementLearning/
├── README.md                       this file
├── requirements.txt                shared dependencies for all projects
├── setup_env.ps1                   Windows PowerShell setup script
├── .gitignore
│
├── AI-Tutor-A2C/
├── digital-calligrapher-rlhf/
├── green-logistics-optimizer/
├── mab-banner-optimizer/
├── market-regime-detector-hmm/
├── marl-warehouse-sim/
├── mbrl-pendulum-playground/
├── rl_maze_solver/
├── rocket-lander-sac/
├── smart-grid-energy-optimizer/
├── swarm-architect-marl/
└── Unity-RL-Huggy-Demo/
```

Each project directory contains its own `app.py`, `README.md`, `requirements.txt`, and module subdirectories.
