<p align="center">
  <img src="assets/banner.png" alt="Reinforcement Learning Lab Banner" width="100%">
</p>

<p align="center">
  <a href="https://dash10107.github.io/reinforcement-learning-lab/en/"><img src="https://img.shields.io/badge/Course-Website-blue?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Course Website"></a>
  <a href="https://huggingface.co/spaces/Dash10107"><img src="https://img.shields.io/badge/Live_Demos-Hugging_Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Demos"></a>
  <a href="https://github.com/Dash10107/reinforcement-learning-lab/stargazers"><img src="https://img.shields.io/github/stars/Dash10107/reinforcement-learning-lab?style=for-the-badge" alt="GitHub stars"></a>
  <a href="https://github.com/Dash10107/reinforcement-learning-lab/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Dash10107/reinforcement-learning-lab?style=for-the-badge" alt="License"></a>
</p>

**Reinforcement Learning Lab** is an open-source, interactive course and project hub designed to bridge the gap between mathematical theory and practical RL implementation. 

Instead of just reading equations or blindly calling high-level APIs, you can learn the concept, inspect the from-scratch code, run the interactive training dashboard, and watch the agent learn in real-time.

[**Explore the Course**](https://dash10107.github.io/reinforcement-learning-lab/en/) • [**Try the Demos**](https://huggingface.co/spaces/Dash10107) • [**Getting Started**](./GETTING_STARTED.md)

---

## 🚀 Quick Start

The fastest way to experience the lab is to try one of the zero-install browser demos, or spin up the code locally.

### 1. Zero-Install Demos
Every project is deployed live. **[Try the Hugging Face Spaces collection →](https://huggingface.co/spaces/Dash10107)**

### 2. Run Locally
Clone the repository and spin up a training lab on your own machine:
```bash
git clone https://github.com/Dash10107/reinforcement-learning-lab.git
cd reinforcement-learning-lab

# Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

# Run a project (e.g., Deep Q-Network for Green Logistics)
cd green-logistics-optimizer
python app.py
```
*This will launch a local dashboard where you can tune hyperparameters and watch the agent train.*

---

## 🗺️ Syllabus & Project Index

Our curriculum progresses from tabular foundations to modern multi-agent and alignment systems. 

| Level | Topic | Project / Implementation | Interactive Demo |
|---|---|---|---|
| **1. Foundations** | Multi-Armed Bandits | [MAB Banner Optimizer](./mab-banner-optimizer) | [▶ Run](https://huggingface.co/spaces/Dash10107/mab-banner-optimizer) |
| | MDP & Tabular RL | [RL Maze Solver](./rl_maze_solver) (Q-Learning/SARSA/MC) | [▶ Run](https://huggingface.co/spaces/Dash10107/rl_maze_solver) |
| **2. Deep RL** | Deep Q-Networks (DQN) | [Green Logistics Optimizer](./green-logistics-optimizer) | [▶ Run](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer) |
| | Discrete Control | [Smart Grid Energy Optimizer](./smart-grid-energy-optimizer) | [▶ Run](https://huggingface.co/spaces/Dash10107/smart-grid-energy-optimizer) |
| **3. Policy Gradients** | Advantage Actor-Critic | [AI Tutor A2C](./AI-Tutor-A2C) | [▶ Run](https://huggingface.co/spaces/Dash10107/AI-Tutor-A2C) |
| | Soft Actor-Critic (SAC) | [Rocket Lander SAC](./rocket-lander-sac) | [▶ Run](https://huggingface.co/spaces/Dash10107/rocket-lander-sac) |
| | PPO (Continuous 3D) | [Unity RL Huggy Demo](./Unity-RL-Huggy-Demo) | [▶ Run](https://huggingface.co/spaces/Dash10107/Unity-RL-Huggy-Demo) |
| **4. Advanced Systems** | Multi-Agent RL (IPPO) | [MARL Warehouse Sim](./marl-warehouse-sim) | [▶ Run](https://huggingface.co/spaces/Dash10107/marl-warehouse-sim) |
| | Cooperative MARL | [Swarm Architect MARL](./swarm-architect-marl) | [▶ Run](https://huggingface.co/spaces/Dash10107/swarm-architect-marl) |
| | Model-Based RL (MPC) | [MBRL Pendulum Playground](./mbrl-pendulum-playground) | [▶ Run](https://huggingface.co/spaces/Dash10107/mbrl-pendulum-playground) |
| | Human Alignment (RLHF) | [Digital Calligrapher RLHF](./digital-calligrapher-rlhf) | [▶ Run](https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf) |
| **Bonus** | Hidden Markov Models | [Market Regime Detector HMM](./market-regime-detector-hmm) | [▶ Run](https://huggingface.co/spaces/Dash10107/market-regime-detector-hmm) |

---

## 📸 Featured Environments

We believe visualization is key to intuition. Our environments are custom-built to expose the inner workings of the algorithms.

<div align="center">
  <img src="assets/previews/rocket_lander.png" width="48%" alt="SAC Rocket Lander">
  <img src="assets/previews/warehouse_robots.png" width="48%" alt="MARL Warehouse">
</div>
<div align="center">
  <img src="assets/previews/maze_solver.png" width="48%" alt="Tabular RL Maze Solver">
  <img src="assets/previews/digital_calligrapher.png" width="48%" alt="RLHF Calligrapher">
</div>

---

## 🤝 Community & Contributing

We welcome contributions! Whether you want to fix a bug, improve documentation, or add an entirely new RL algorithm or environment, please feel free to fork the repository.

1. Check out our [Contribution Guidelines](./CONTRIBUTING.md).
2. Grab an open issue or suggest a new feature.
3. Build, experiment, and open a PR!

If this repository helped you learn, consider leaving a ⭐ **Star** so others can find it!

## 📜 License

This project is licensed under the [MIT License](./LICENSE).
