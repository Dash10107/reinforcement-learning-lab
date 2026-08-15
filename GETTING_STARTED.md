# Getting Started

This guide walks you through running your first project from this portfolio — step by step, with exactly what to expect at each stage. Follow it before trying anything else.

---

## What you need

- Python 3.10 or 3.11 (3.12 works but some packages lag behind)
- Git
- A terminal — PowerShell on Windows, Terminal on Mac or Linux
- About 2 GB of free disk space for the packages

Check your Python version:

```bash
python --version
```

If you see 3.10, 3.11, or 3.12 you are good. If you see 2.x or 3.9 or lower, install a newer version from python.org.

---

## Step 1 — Get the code

```bash
git clone https://github.com/Dash10107/reinforcement-learning-lab
cd reinforcement-learning-lab
```

Or if you are working directly from the folder on your machine, just open a terminal and navigate to the `ReinforcementLearning` directory.

---

## Step 2 — Set up the environment

**Windows (PowerShell):**

```powershell
.\setup_env.ps1
```

**Mac or Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs everything needed for all 12 projects into a single virtual environment. It will take a few minutes the first time.

**Activate the environment before running any project:**

```powershell
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

You will see `(.venv)` at the start of your terminal prompt when it is active.

---

## Step 3 — Run your first project

Start with the RL Maze Solver. It is the simplest project, runs fast, and makes the core RL loop immediately visible.

```bash
cd rl-maze-solver
python app.py
```

You should see output like:

```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser.

---

## Step 4 — What you are looking at

The app opens on the Welcome tab. Read the three cards — they explain what the maze, the robot, and the learning process are in plain terms.

Go to the **Playground** tab.

- Leave the difficulty on **Medium (9×9)**
- Leave the algorithm on **Q-Learning**
- Set episodes to **500**
- Click **Train and Watch**

Wait about 10 seconds. You will see:

1. An animated GIF of the agent walking through the maze — this is the learned policy after 500 episodes
2. A training chart showing reward per episode — it should trend upward from very negative toward zero
3. A Q-value heatmap showing which cells the agent has learned to value — brighter cells are on or near the optimal path

---

## Step 5 — Your first experiment

Change one thing and observe what happens.

**Try this:** Change the difficulty to **Large (13×13)** and run again with the same 500 episodes.

What you should see: the training curve improves less, the path is longer, and the Q-value heatmap is patchier. A larger maze has more states to learn — 500 episodes is not enough. Try 1500 episodes and run again.

This is your first hands-on observation of **sample efficiency** — how many interactions an agent needs to learn a good policy. Different algorithms have very different sample efficiency. The Algorithm Race tab lets you compare them directly.

---

## Step 6 — Try the Algorithm Race

Still in the Maze Solver app, go to the **Algorithm Race** tab.

- Set difficulty to **Medium (9×9)**
- Set episodes to **600**
- Tick **Include Monte Carlo**
- Click **Start Race**

Three algorithms train on the same maze simultaneously. The convergence chart shows which one learns fastest and which one reaches the best final performance. Monte Carlo tends to converge slowly but ends up accurate. Q-Learning converges fast. SARSA is in between but more cautious near walls.

Understanding these tradeoffs is the first step to knowing which algorithm to reach for in a new problem.

---

## Step 7 — Explore the other projects

Once you are comfortable with the maze solver, these are the best next steps depending on what interests you.

**If you want to understand how neural networks enter RL:**
Go to the Green Logistics Optimizer. This is DQN — Q-Learning but with a neural network instead of a table. The city map makes it visually clear what the agent has learned.

**If you want to understand policy gradient methods:**
Go to the AI Tutor A2C. The Analyse tab shows the probability distribution the policy assigns to each subject — you can see the neural network's "thinking" directly.

**If you want to see continuous control:**
Go to the Rocket Lander SAC. Set wind to enabled, run 3 landing attempts, and watch the animated GIF with the throttle HUD overlay.

**If you want to see multiple agents:**
Go to the Swarm Architect MARL or MARL Warehouse Sim. The benchmark tab runs trained vs untrained strategies side by side so the difference is immediately visible.

**If you want to understand real-world applications:**
Go to the Smart Grid Energy Optimizer or the Market Regime Detector. These are the least game-like projects — the closest to what you would actually deploy in industry.

---

## Common problems

**The app does not start:**
Make sure the virtual environment is activated. You should see `(.venv)` in your terminal. If not, run the activate command from Step 2.

**ImportError or ModuleNotFoundError:**
The package is not installed. Run `pip install -r requirements.txt` again inside the virtual environment.

**Port 7860 is already in use:**
Another Gradio app is running. Either close it or pass a different port: `python app.py --server-port 7861`

**Training seems stuck:**
It is not stuck — it is running. RL training is compute-intensive. A 500-episode maze run takes about 5-10 seconds. A 20,000-step DQN training takes about 30-60 seconds. A 200-episode MARL run takes about 60-90 seconds.

**The animated GIF is blank or not loading:**
Some browsers block autoplay GIFs. Try Firefox or disable content restrictions for localhost.

**ModuleNotFoundError: No module named 'mpe2':**
This is specific to the Swarm Architect project. Install it separately: `pip install mpe2`

---

## Running on the live demos without installing anything

Every project is deployed on Hugging Face Spaces. If you just want to explore without setting up a local environment, go to:

```
https://huggingface.co/spaces/Dash10107
```

All 12 projects are accessible from there. Note that the free tier on HuggingFace has limited compute — training runs will be slower than local.

---

## What to read next

After running a few projects, go back and read the individual project READMEs more carefully. They explain the algorithm, the environment dynamics, and what each chart means. Once you have seen the app running, the explanations will make much more sense.

If you want a deeper understanding of the algorithms themselves, the `CONCEPTS.md` file in this directory is the right next read.
