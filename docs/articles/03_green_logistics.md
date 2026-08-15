---
title: "Planning vs Reacting: Why A* Search Fails in Traffic, and How Deep Q-Networks Fix It"
published: true
description: "A deep dive into why classical pathfinding algorithms fail in dynamic environments, and how Deep Q-Networks use Experience Replay and Target Networks to build real-time routing intuition."
tags: "machinelearning, python, reinforcementlearning, ai, logistics"
series: "Reinforcement-Learning"
cover_image: "https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/logistics_cover.png"
---

![Green Logistics Cover](https://raw.githubusercontent.com/Dash10107/reinforcement-learning-lab/main/assets/logistics_cover.png)

Imagine you are a delivery driver navigating a busy city. Your goal is to get from Point A to Point B. If you open a standard navigation app, it will likely use a variation of a classical search algorithm—like Dijkstra's or A* (A-Star)—to draw a line showing the absolute shortest path.

But any experienced driver knows a fundamental truth about city driving: **the shortest path is rarely the cheapest path.**

In modern logistics, companies aren't just optimizing for distance; they are optimizing for fuel consumption and carbon emissions. Heavy traffic zones multiply a vehicle's carbon output (and fuel cost) by a factor of four. An experienced human driver intuitively learns to take a slightly longer, winding route to completely bypass a congested downtown core. 

To teach an AI to develop this exact same intuition, I built a **[Green Logistics Optimizer](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer)**. In this interactive simulation, we pit a classical mathematical planner (A*) against a Deep Reinforcement Learning agent (DQN). 

In this article, we're going to dive into the underlying theory of why classical planning algorithms eventually hit a wall in the real world, and how Neural Networks solve the problem by learning to *react* rather than *plan*.

---

## 1. The Flaw of the Perfect Planner (A*)

Let's start with the baseline. The A* algorithm is a masterpiece of computer science. It guarantees finding the optimal path between two points by using a "heuristic"—an educated guess of the remaining distance.

In our Python implementation, A* evaluates the grid by adding the distance traveled so far (`g`) to the estimated distance to the goal (`h`). It uses a Priority Queue (a heap) to always explore the most promising path first:

```python
def _astar_path(env: GreenCityEnv):
    # A* minimises step count (distance), completely ignoring carbon.
    start = tuple(env.agent_pos.tolist())
    goal  = tuple(env.goal.tolist())

    # The Heuristic: Manhattan distance to the goal
    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap = [(h(start), 0, start, [start])]
    visited = set()

    while open_heap:
        _, g, cur, path = heapq.heappop(open_heap)
        
        if cur == goal: return path
        # ... explore neighbors, add to heap, repeat ...
```

If you look at this code, you'll notice it perfectly calculates the shortest physical distance. 

**So, what's the problem?**
The problem is that A* is a *planner*. It requires you to know the entire map, and the exact cost of every street, in advance. If you try to modify A* to include "traffic costs" instead of just distance, it works fine—*until the traffic changes*. 

If a traffic light turns red, or an accident occurs, the weights of your map change. A* has to throw away its entire planned route and recalculate the mathematical tree from scratch. In a massive city grid with millions of nodes and dynamically shifting traffic, constantly recalculating A* becomes computationally paralyzing.

We don't want an AI that mathematically calculates a billion possibilities every time a car brakes. We want an AI that looks at the traffic and just *knows* what to do.

---

## 2. The Shift: From Planning to Reacting

This is where Reinforcement Learning enters the picture. Instead of writing an algorithm that searches a map, we drop an agent into the city and let it drive around millions of times. 

We use a **Deep Q-Network (DQN)**. 
Unlike A*, a DQN doesn't plan a route from start to finish. Instead, it looks at a snapshot of the current state (its location and the traffic around it) and outputs a "Q-Value" for all possible immediate actions (Up, Down, Left, Right). 

The Q-Value represents the *expected future reward* of taking that action, mathematically defined by the **Bellman Optimality Equation**:

**Q(s, a) = Reward(s, a) + γ * max Q(s_next, a_all)**

Because the environment is too massive for a simple lookup table, we use a Neural Network as a mathematical function approximator to estimate these Q-Values. In PyTorch, the "Brain" of the vehicle looks like this:

```python
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim) # Outputs 4 Q-Values (Up, Down, Left, Right)
        )
        
    def forward(self, state):
        return self.network(state)
```

The agent simply passes its current state into this network, gets the four numbers, and picks the highest one. It's a purely reactive system.

```python
def run_dqn(env: GreenCityEnv) -> RouteResult:
    # Load the trained Neural Network
    model = DQN.load("green_dqn_model", env=env)

    def _dqn_action(pos, env):
        # Look at the current state, instantly predict the best move
        action, _ = model.predict(pos, deterministic=True)
        return int(action)
        
    return _rollout(env, _dqn_action)
```

Notice there is no `while` loop searching through a map. There is just a single `model.predict()` call. Because the neural network has already "compiled" the knowledge of the city into its weights during training, querying it takes milliseconds.

But training a neural network to do this requires overcoming two fascinating theoretical hurdles.

---

## 3. Overcoming Catastrophic Forgetting (Experience Replay)

Neural Networks have a fatal flaw when used in Reinforcement Learning: **Catastrophic Forgetting**.

Imagine our delivery driver gets stuck driving around a heavy congestion zone for 500 consecutive steps. The neural network is constantly updating its weights based on this high-traffic data. Because neural networks generalize, the math that adjusts the weights for the traffic zone will aggressively overwrite the weights that control how the car drives on an open, empty highway. By the time the agent escapes the traffic, it has literally "forgotten" how to drive in an empty street!

### The Solution: Experience Replay
To solve this, DQN introduced a brilliant concept called **Experience Replay**. 

Instead of training the neural network on the exact sequence of events as they happen, the agent takes every single step it makes and throws it into a massive database called a Replay Buffer. 

When it's time to train, the network doesn't look at what just happened. Instead, it reaches into the Replay Buffer and pulls out a completely randomized mini-batch of past experiences:

```python
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.memory = []
        
    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def dream(self, batch_size=64):
        # Pull a randomized batch of 64 memories to break correlation
        return random.sample(self.memory, batch_size)
```

It might pull one memory from a traffic jam, one memory from an empty highway, and one memory of reaching the goal. By shuffling its memories (`random.sample`), the neural network mathematically breaks the correlation between consecutive steps. It is the artificial equivalent of human REM sleep—dreaming and shuffling past experiences to consolidate generalized, permanent knowledge.

---

## 4. Chasing a Moving Target (Target Networks)

If you've ever trained a standard image classifier, you know the data has fixed labels. An image of a cat is always a cat. The network predicts "Dog", calculates the error against the fixed label "Cat", and updates.

But in Deep Q-Networks, the network is trying to predict a Q-Value, and the "correct label" (the Target) is generated by the Bellman Equation we discussed earlier.

Do you see the paradox? If we use a single neural network, the Target is being generated by the *exact same neural network* that we are currently updating! If the network adjusts its weights to increase the value of moving "Up", it accidentally changes the calculated target for the next step too. It is like a dog violently chasing its own tail. The network weights will oscillate wildly and mathematically explode.

### The Solution: Dual Networks
To stabilize the math, we maintain two identical Neural Networks in PyTorch:
1. **The Policy Network:** Actively driving the car and updating its weights every step.
2. **The Target Network:** A completely frozen clone of the Policy Network.

When calculating the "correct label", we ask the frozen Target Network:

```python
# 1. Ask the FROZEN Target Network for the next state's value
with torch.no_grad():
    target_next_q = target_network(next_state).max(1)[0]
    
# 2. Calculate the perfectly stable target label
target_label = reward + (gamma * target_next_q * (1 - done))

# 3. Train the ACTIVE Policy Network against this frozen label
current_q = policy_network(state).gather(1, action)
loss = F.mse_loss(current_q, target_label)
```

Because the Target Network is frozen (`torch.no_grad()`), the mathematical anchor stays perfectly still, allowing the Policy Network to steadily learn. Every few hundred steps, we take the weights from the Policy Network and explicitly copy them over to the Target Network (`target_network.load_state_dict(policy_network.state_dict())`). This simple trick is the theoretical bedrock that makes Deep RL stable.

---

## 5. The Power of the Reward Function

When you put all of this together, something magical happens. The agent's behavior is dictated entirely by a tiny reward function. In our environment:

*   **Diesel Truck:** `Base Cost = 1.0`. Congestion Multiplier = `4.0`. 
*   **Electric Vehicle (EV):** `Base Cost = 0.2`. Congestion Multiplier = `4.0`.

A diesel truck entering congestion suffers a massive `-4.0` reward penalty per step. An EV suffers a `-0.8` penalty.

When you train the DQN, you don't have to write any complex `if/else` statements telling the Diesel truck to avoid traffic. The math does it automatically. The Diesel agent will learn to take a massive detour around the city to avoid traffic, while the EV agent might decide it's mathematically cheaper to just cut straight through the congestion because its base emissions are so low.

---

## 🧪 Try It Yourself

To truly understand the difference between planning and reacting, you have to see the visual traces. Open up the **[Green Logistics Simulator](https://huggingface.co/spaces/Dash10107/green-logistics-optimizer)** and run these experiments:

1. **The Distance vs Carbon Gap:** Pick the `Downtown Rush (7x7)` scenario. Select both A* and DQN. Click Deploy Fleet. You will see the A* path cut straight through the red congestion zone (because it is the shortest path). But look at the DQN path—it curves widely around the red zone to save carbon.
2. **Check the Analytics:** Go to the Analytics tab. Look at the Carbon Trace. A* reaches the goal in fewer steps, but its cumulative carbon spikes violently. DQN takes more steps, but its carbon line stays flat and low.
3. **Train Your Own Brain:** Go to the Training Lab. Set the slider to `10,000` steps and click Train. You can watch the live reward curve rise as the network populates its Replay Buffer and slowly stabilizes its Target Network.

---

### Wrapping Up

Classical algorithms like A* are beautiful, but they require perfect, static knowledge of the world to plan ahead. By utilizing Deep Q-Networks, Experience Replay, and Target Networks, we can teach AI to simply look at a chaotic, shifting environment and intuitively *react*.

This is the third of 12 interactive RL projects I am building to bridge the gap between academic math and real-world intuition. If this deep dive helped clarify the theory inside Neural Networks, I would be incredibly grateful if you checked out the source code and dropped a star on the full repository:

⭐ **Reinforcement Learning Lab on GitHub**{% embed https://github.com/Dash10107/reinforcement-learning-lab %}

Let me know in the comments: *Which feels more "intelligent" to you—a mathematical algorithm that plans perfectly, or a neural network that guesses intuitively?*
