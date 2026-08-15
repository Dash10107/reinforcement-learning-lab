---
title: Green Logistics Optimizer
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Green Logistics Optimizer — DQN Delivery Route Planning

A reinforcement learning system that learns to plan urban delivery routes while minimising carbon emissions. A Deep Q-Network agent navigates a city grid, learning to avoid high-congestion zones that multiply fuel costs. You can compare the DQN agent against a greedy heuristic and an A* shortest-path solver, visualise their routes on a carbon heatmap, and retrain the agent from scratch.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer)

---

## What problem does this solve?

A delivery driver wants to get from point A to point B as fast as possible. But in a city with traffic congestion, the shortest route is not always the cheapest route in terms of fuel or emissions. Heavy-traffic zones multiply your carbon output by a factor of four. An experienced driver learns over time to choose slightly longer routes that avoid those zones, especially if driving a diesel vehicle where the cost difference is large.

This project trains an RL agent to discover that same intuition by interacting with a simulated city, receiving rewards for low-emission deliveries and penalties for high-emission routes.

---

## The Algorithm: Deep Q-Network (DQN)

DQN is an off-policy algorithm that learns a Q-function, a mapping from (state, action) pairs to expected future rewards. The agent always picks the action with the highest Q-value in its current state. Training updates the Q-function using the Bellman equation:

    Q(state, action) = reward + gamma * max Q(next_state, all_actions)

This says: the value of taking an action should equal the immediate reward plus the discounted value of the best action available from the next state.

What makes DQN different from basic Q-learning is that the Q-function is represented by a neural network rather than a lookup table. This lets it generalise across states it has not seen exactly before. Two stability tricks make this work in practice.

Experience Replay: Rather than learning from each experience immediately, DQN stores transitions in a replay buffer and samples random mini-batches for training. This breaks the correlation between consecutive experiences that would otherwise destabilise training.

Target Network: DQN keeps a frozen copy of the Q-network used to compute training targets. This network updates slowly every few hundred steps. Without this, the targets shift constantly as the network trains, causing oscillation.

---

## The Environment

The city is modelled as an N by N grid, default 7 by 7. The agent starts at one corner and must reach a goal position.

State: The agent current position as a (row, column) pair.

Actions: Four directions, up, down, left, right.

Reward per step:

    base_cost   = 1.0  for Diesel  or  0.2  for Electric Vehicle
    multiplier  = 4.0  if in a congestion zone,  else  1.0
    step_reward = -(base_cost * multiplier)
    delivery    = +20  on reaching the goal

A diesel vehicle crossing a congestion zone pays a cost of 4.0 for that step. An EV in the same zone pays only 0.8. Both have an incentive to avoid congestion, but the incentive is much stronger for diesel.

Episode end: The episode ends when the agent reaches the goal or after (grid_size * 5) steps maximum.

---

## Three Strategies Compared

The app runs all three strategies on the same city so you can see the difference directly.

DQN Agent: The trained neural network. It has learned from thousands of episodes that congestion zones are expensive and tries to route around them when doing so is worth the extra steps.

Greedy Heuristic: At every step, move one cell closer to the goal using Manhattan direction. It never detours. It crosses congestion zones whenever they sit on the direct path.

A* Optimal: Finds the path with the fewest steps using the A* search algorithm. This is the shortest route by distance, but it ignores emission costs entirely. A* can still beat greedy because it avoids backtracking and dead ends.

The key distinction is that A* minimises steps but not carbon. The DQN agent minimises carbon, which sometimes means taking more steps. On a city with dense congestion clusters, the DQN route can be significantly cleaner even if it is a few steps longer.

---

## Project Structure

    green-logistics-optimizer/
    |-- app.py                  main Gradio application
    |-- core/
    |   |-- env.py              GreenCityEnv custom Gymnasium environment
    |   |-- agents.py           DQN, Greedy, A* implementations and training
    |-- viz/
    |   |-- charts.py           city heatmap, carbon trace, performance radar
    |-- requirements.txt

---

## Quick Setup

Clone and install:

    git clone https://github.com/yourusername/reinforcement-learning-lab
    cd green-logistics-optimizer
    pip install -r requirements.txt

Run:

    python app.py

Open http://localhost:7860 in your browser.

To try it out, pick a scenario preset from the dropdown. Downtown Rush is a good starting point. Make sure all three strategy checkboxes are ticked and click Deploy Fleet. The city carbon map will appear with all three routes overlaid. Then go to the Analytics tab for the carbon trace and performance radar.

To train your own DQN agent, go to the Training Lab tab, set the timestep budget, and click Start Training. The model saves automatically and loads the next time you run the DQN strategy.

---

## What each tab shows

Mission Control: Configuration panel with scenario preset, grid size, start and goal positions, vehicle type, and congestion zone coordinates. Click Deploy Fleet to run all selected strategies and see the city map with routes overlaid, plus a summary table showing carbon totals, steps, and congestion hits per strategy.

Analytics: Carbon trace chart showing cumulative emissions step by step for each strategy, a total carbon bar chart, and a performance radar with five normalised metrics per strategy: Eco Score, Speed, Safety, Efficiency, and Delivery.

Training Lab: Train a DQN from scratch with a live reward curve showing episode returns during training. The model saves to disk and is used automatically in Mission Control.

How DQN Works: Full explanation of the Bellman equation, experience replay, target networks, the carbon cost formula, and how to read each chart.

---

## Scenario Presets

Downtown Rush (7x7): Dense congestion cluster in the urban core.
Industrial Port (8x8): Congestion near loading docks on two sides.
Ring Road (7x7): Congestion at junction corners with a clear central corridor.
Open Field (6x6): No congestion, useful as a baseline.

---

## Requirements

    gradio>=6.0.0
    gymnasium
    numpy
    matplotlib
    stable-baselines3
    shimmy

---

## Things to Try

**1. Run the benchmark on Downtown Rush and note the carbon gap.**
How much more carbon does Greedy emit compared to A*? How close does DQN get to A*? The gap between DQN and A* is the amount of learning still remaining.

**2. Switch from Diesel to EV and re-run.**
Absolute carbon numbers drop but the relative comparison between strategies stays similar. The learned route avoidance is independent of vehicle type.

**3. Place congestion zones directly on the shortest path.**
Use the custom congestion text input. Greedy will walk straight through them. DQN (if trained) should route around them. A* ignores emission cost entirely.

**4. Train for 5,000 steps then benchmark, then 30,000 then benchmark again.**
Carbon savings percentage should improve with more training. This makes sample efficiency tangible.

**5. Try the Open Field scenario with EV.**
With no congestion and an EV all three strategies perform almost identically. This shows the DQN congestion-avoidance skill only matters when there is congestion to avoid.

---

## Further Reading

Mnih et al., Human-level control through deep reinforcement learning (2015). The original DQN paper.

Stable-Baselines3 DQN documentation: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

Hart et al., A Formal Basis for the Heuristic Determination of Minimum Cost Paths (1968). The A* algorithm.
