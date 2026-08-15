<p align="center">
  <img src="assets/banner.png" alt="Reinforcement Learning Lab — Interactive RL Course and Projects" width="900" />
</p>

<p align="center">
  <a href="https://dash10107.github.io/reinforcement-learning-lab/en/">
    <img src="https://img.shields.io/badge/🌐_Course-Visit_the_Lab-blue?style=for-the-badge" alt="Visit Reinforcement Learning Lab" />
  </a>
  <a href="https://huggingface.co/spaces/Dash10107">
    <img src="https://img.shields.io/badge/🤗_Demos-Hugging_Face-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face demos" />
  </a>
  <a href="https://github.com/Dash10107/reinforcement-learning-lab/stargazers">
    <img src="https://img.shields.io/github/stars/Dash10107/reinforcement-learning-lab?style=for-the-badge" alt="GitHub stars" />
  </a>
  <a href="https://github.com/Dash10107/reinforcement-learning-lab/network/members">
    <img src="https://img.shields.io/github/forks/Dash10107/reinforcement-learning-lab?style=for-the-badge" alt="GitHub forks" />
  </a>
</p>

<h1 align="center">Reinforcement Learning Lab</h1>

<p align="center">
  <strong>Learn Reinforcement Learning by Building.</strong>
</p>

<p align="center">
  An open-source, interactive reinforcement learning course and project lab —
  from multi-armed bandits and Q-Learning to DQN, PPO, SAC, multi-agent RL,
  model-based RL, and RLHF.
</p>

<p align="center">
  <strong>Read the idea → understand the math → inspect the code → run the agent → change the parameters → see what happens.</strong>
</p>

---

## 🧭 Why this repository exists

Reinforcement learning is often taught in one of two ways:

1. **Theory first** — equations, Bellman updates, policy gradients, proofs.
2. **Library first** — call a high-level API, train an agent, and hope it becomes intuitive.

**Reinforcement Learning Lab bridges the gap.** Every project is designed so that you can move through the complete loop:

**`Concept → Intuition → Math → Code → Experiment → Modify`**

The goal is not just to show that an agent can learn, but to help you understand *why* it learns, *when* it fails, and *what changes* when you alter the environment.

---

## ✨ What's Inside

- **12 end-to-end RL projects** covering foundational, deep, policy-gradient, multi-agent, model-based, and alignment settings.
- **Interactive browser demos** hosted on Hugging Face Spaces.
- **A structured curriculum** rather than an unconnected collection of notebooks.
- **Visual experiments** with learning curves, trajectories, value maps, and policy behaviour.
- **From-scratch implementations** alongside practical deep-RL tooling.
- **Browser-based development** via GitHub Codespaces + Google Colab.

---

## 🗺️ Learning Path

The lab is designed to progress from simple tabular problems to modern RL systems. You can read the full course material on the **[Official Course Website](https://dash10107.github.io/reinforcement-learning-lab/en/)**.

**Foundations** → **Value-Based (DQN)** → **Actor-Critic (PPO/SAC)** → **Multi-Agent (MARL)** → **Model-Based (MBRL)** → **Alignment (RLHF)**

---

## 🚀 Featured Projects

<table width="100%">
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">🤖 RL Maze Solver</h3>
      <a href="https://huggingface.co/spaces/Dash10107/rl_maze_solver">
        <img src="assets/previews/maze_solver.png" width="100%" alt="Interactive reinforcement learning maze solver" />
      </a>
      <p>Compare Q-Learning, SARSA, and Monte Carlo while watching agents learn to solve procedurally generated mazes. Inspect convergence curves and Q-value heatmaps.</p>
      <p align="center">
        <a href="./rl_maze_solver"><strong>📂 Source</strong></a> · <a href="https://huggingface.co/spaces/Dash10107/rl_maze_solver"><strong>🚀 Live Demo</strong></a>
      </p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">🚀 Rocket Lander SAC</h3>
      <a href="https://huggingface.co/spaces/Dash10107/rocket-lander-sac">
        <img src="assets/previews/rocket_lander.png" width="100%" alt="Soft Actor-Critic reinforcement learning rocket landing demo" />
      </a>
      <p>Train a Soft Actor-Critic agent to control a rocket in continuous action space. Experiment with wind and gravity, inspect telemetry, and fine-tune in the browser.</p>
      <p align="center">
        <a href="./rocket-lander-sac"><strong>📂 Source</strong></a> · <a href="https://huggingface.co/spaces/Dash10107/rocket-lander-sac"><strong>🚀 Live Demo</strong></a>
      </p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">📦 MARL Warehouse Sim</h3>
      <a href="https://huggingface.co/spaces/Dash10107/marl-warehouse-sim">
        <img src="assets/previews/warehouse_robots.png" width="100%" alt="Multi-agent reinforcement learning warehouse simulation" />
      </a>
      <p>Multiple robots learn to coordinate package deliveries in a custom warehouse environment using Independent PPO. Observe task allocation and collision avoidance.</p>
      <p align="center">
        <a href="./marl-warehouse-sim"><strong>📂 Source</strong></a> · <a href="https://huggingface.co/spaces/Dash10107/marl-warehouse-sim"><strong>🚀 Live Demo</strong></a>
      </p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">🎨 Digital Calligrapher RLHF</h3>
      <a href="https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf">
        <img src="assets/previews/digital_calligrapher.png" width="100%" alt="Interactive RLHF preference learning calligrapher demo" />
      </a>
      <p>Vote between generated brush strokes, learn a preference model from those choices, and optimise new outputs toward your learned aesthetic preferences.</p>
      <p align="center">
        <a href="./digital-calligrapher-rlhf"><strong>📂 Source</strong></a> · <a href="https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf"><strong>🚀 Live Demo</strong></a>
      </p>
    </td>
  </tr>
</table>

---

## 🔬 All Projects Index

| Project | Method | What you learn | Demo |
|---|---|---|---|
| [AI Tutor A2C](./AI-Tutor-A2C) | A2C | State values, advantages, policy learning | [▶ Demo](https://huggingface.co/spaces/Dash10107/AI-Tutor-A2C) |
| [Digital Calligrapher RLHF](./digital-calligrapher-rlhf) | RLHF + Bradley-Terry | Preference learning and reward modelling | [▶ Demo](https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf) |
| [Green Logistics Optimizer](./green-logistics-optimizer) | DQN | Experience replay, target networks, routing | [▶ Demo](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer) |
| [MAB Banner Optimizer](./mab-banner-optimizer) | Bandits | Exploration, regret, Bayesian/adversarial strategies | [▶ Demo](https://huggingface.co/spaces/Dash10107/mab-banner-optimizer) |
| [Market Regime Detector HMM](./market-regime-detector-hmm) | Gaussian HMM | Hidden states, Viterbi, Baum-Welch | [▶ Demo](https://huggingface.co/spaces/Dash10107/market-regime-detector-hmm) |
| [MARL Warehouse Sim](./marl-warehouse-sim) | Independent PPO | Multi-agent coordination and collision avoidance | [▶ Demo](https://huggingface.co/spaces/Dash10107/marl-warehouse-sim) |
| [MBRL Pendulum Playground](./mbrl-pendulum-playground) | Ensemble + MPC | Learned dynamics, uncertainty, planning | [▶ Demo](https://huggingface.co/spaces/Dash10107/mbrl-pendulum-playground) |
| [RL Maze Solver](./rl_maze_solver) | Q-Learning/SARSA/MC | Foundational value-based RL | [▶ Demo](https://huggingface.co/spaces/Dash10107/rl_maze_solver) |
| [Rocket Lander SAC](./rocket-lander-sac) | SAC | Entropy, actor-critic, continuous control | [▶ Demo](https://huggingface.co/spaces/Dash10107/rocket-lander-sac) |
| [Smart Grid Energy Optimizer](./smart-grid-energy-optimizer) | DQN + DP | Sequential decision-making under dynamic prices | [▶ Demo](https://huggingface.co/spaces/Dash10107/smart-grid-energy-optimizer) |
| [Swarm Architect MARL](./swarm-architect-marl) | Independent PPO | Cooperative landmark coverage | [▶ Demo](https://huggingface.co/spaces/Dash10107/swarm-architect-marl) |
| [Unity RL Huggy Demo](./Unity-RL-Huggy-Demo) | PPO (Unity ML-Agents) | Sparse rewards and 3D continuous control | [▶ Demo](https://huggingface.co/spaces/Dash10107/Unity-RL-Huggy-Demo) |

---

## 💻 Quick Start

### 1. Try the live demos
No setup required. **[🤗 Open the Hugging Face demo collection](https://huggingface.co/spaces/Dash10107)**.

### 2. Open in GitHub Codespaces
Run the code instantly in your browser. **[☁️ Open in Codespaces](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=Dash10107/reinforcement-learning-lab)**.

### 3. Run Locally

```bash
git clone https://github.com/Dash10107/reinforcement-learning-lab.git
cd reinforcement-learning-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Then `cd` into any project folder and run `python app.py`. See [GETTING_STARTED.md](./GETTING_STARTED.md) for detailed instructions.

---

## 🤝 Contributing

This project is intended to grow into a shared learning resource. Whether it's fixing an implementation, adding an experiment, or writing documentation, contributions are highly welcome. 

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a pull request.

---

## ⭐ Support the Lab

If this repository helps you learn, build, teach, or experiment with reinforcement learning:

- **⭐ Star the repository** so you can find it again.
- **🍴 Fork it** and build your own experiments.
- **📢 Share it** with someone learning RL.

---

## 📜 License

Released under the [MIT License](./LICENSE).
