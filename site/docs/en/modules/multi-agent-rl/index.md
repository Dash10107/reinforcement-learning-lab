---
description: "Multi-Agent Reinforcement Learning (MARL) explained. Covers IPPO, Nash equilibrium, non-stationarity, CTDE architecture, and QMIX monotonic mixing for cooperative settings."
---

# What Happens When There Are Two of You?

Everything we've built so far has one agent, one environment.

The agent acts. The environment responds. Only the agent is learning.

Now imagine two agents in the same environment. Both are learning. Both are reacting to each other. When Agent A improves its policy, the world that Agent B sees changes — because Agent A is now behaving differently.

This is **multi-agent reinforcement learning (MARL)**, and it introduces a problem that doesn't exist in single-agent RL: **non-stationarity**.

---

## The non-stationarity problem

In single-agent RL, the environment follows fixed rules. The maze doesn't change. The physics doesn't change. The agent is the only thing that changes.

In multi-agent RL, other agents are part of the environment. And they're learning — which means the environment *is* changing. From any single agent's perspective, it looks like the rules are shifting under its feet.

This breaks the theoretical guarantees that most single-agent RL algorithms rely on. Q-learning converges when the environment is stationary. In a multi-agent setting, it's not.

---

## IPPO: the simplest approach that often works

Given how hard the general problem is, the most widely-used approach in practice is also the most conceptually simple: **Independent PPO (IPPO)**.

Each agent runs its own PPO algorithm. Each agent has its own actor and critic. Each agent treats everything else in the environment — including other agents — as part of the environment. No coordination mechanism. No shared parameters. Just independent learners.

```python
# One agent's training loop (identical for all agents)
for agent_id in range(n_agents):
    # Each agent sees its own local observation
    obs = observations[agent_id]

    # Each agent has its own actor and critic
    action, log_prob = actors[agent_id].sample(obs)
    value = critics[agent_id](obs)

    # Each agent updates using its own experience
    ppo_update(actors[agent_id], critics[agent_id], agent_trajectories[agent_id])
```

The surprising thing: this works. Not always, and not optimally — but in many cooperative settings, independent agents that each optimise their own reward end up cooperating anyway, because cooperation is often the best individual strategy.

In our warehouse environment, robots that learn to avoid each other get more packages delivered individually — so the cooperative behaviour emerges from individual self-interest, not from any explicit coordination mechanism.

---

## Nash equilibrium: when no one wants to change

In game theory, a **Nash equilibrium** is a situation where no player can improve their outcome by unilaterally changing their strategy. In multi-agent RL, it's the natural solution concept:

$$\forall i, \forall a_i': \quad V^{\pi_i, \pi_{-i}}(s) \geq V^{\pi_i', \pi_{-i}}(s)$$

Where:
- $\pi_i$ = agent $i$'s policy
- $\pi_{-i}$ = the policies of all other agents
- $V^{\pi_i, \pi_{-i}}$ = agent $i$'s value when everyone plays their current policy
- The equation says: no agent can do strictly better by switching their policy, given everyone else stays the same

In plain English: *a Nash equilibrium is a stable resting point — nobody has a reason to deviate.*

IPPO doesn't guarantee convergence to a Nash equilibrium. But in cooperative settings (shared reward), it often finds one anyway — because the optimal joint policy is usually also a Nash equilibrium.

---

## CTDE: centralised training, decentralised execution

IPPO treats other agents as part of the environment. A more powerful approach is **Centralised Training, Decentralised Execution (CTDE)**:

- **During training:** use global information — all agents' observations and actions — to train the critics. Better estimates, faster convergence.
- **During execution:** each agent acts using only its own local observation. Scalable, no communication required.

This is the design philosophy behind **MAPPO** (Multi-Agent PPO with shared global critic) and **QMIX**.

**QMIX** takes this further for cooperative settings. Instead of independent Q-values, it learns a **mixing network** that combines all agents' individual Q-values into a joint Q-value:

$$Q_{\text{tot}}(\tau, a) = f_{\phi}(Q_1(\tau_1, a_1), \ldots, Q_n(\tau_n, a_n))$$

With the constraint that $\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0$ for all $i$ — each individual Q-value has a non-negative influence on the joint value. This monotonicity constraint ensures that the globally optimal joint action can be found by each agent locally maximising their own Q-value.

In plain English: *agents coordinate implicitly through a learned mixing function, without ever needing to communicate at runtime.*

---

## What changes in multi-agent settings

**Observations become local.**
In a single-agent maze, the agent might see the full grid. In a multi-agent warehouse, each robot only sees what's near it. The full state of the environment includes what all other robots are doing — but each agent only receives a piece of it.

**Rewards can be individual or shared.**
In a fully cooperative setting, all agents share one reward — the total packages delivered. In a competitive setting, one agent's gain is another's loss. In mixed settings, each agent has its own reward that partially overlaps with others'.

**The action space scales.**
With N agents, the joint action space is the Cartesian product of all individual action spaces. For 4 agents each with 5 actions, that's 5⁴ = 625 possible joint actions. IPPO avoids this by having each agent act independently — they never see or reason about the joint action space.

---

## Cooperation that nobody planned

The most fascinating thing in multi-agent RL is when agents develop coordination strategies that nobody programmed.

In the warehouse demo, the robots learn traffic patterns. They develop informal "lanes" — not because anyone told them to, but because collisions slow them down and they individually learn to avoid the paths where other robots tend to be.

This is a simple version of an idea that runs deep in multi-agent AI: **emergent coordination**. Complex group behaviour arising from simple individual policies. We'll see a much more striking version of this in the next chapter.

---

## What you'll notice in the demo

Open the [Warehouse Robots ↗](https://huggingface.co/spaces/Dash10107/marl-warehouse-sim) — 4 robots on a 12×12 grid picking up and delivering packages.

**Three things to watch:**

1. **Early episodes.** Robots constantly collide and block each other. Throughput is low. Each agent is exploring independently — they have no model of each other.

2. **Traffic patterns forming.** Around episode 500–1000, informal paths emerge. Watch robots start to avoid the centre of the grid during peak traffic. Nobody told them to. They learned it.

3. **Bottleneck behaviour.** Watch what happens at the drop-off point. Early: robots queue and block each other. Late: a natural queuing behaviour develops. One robot waits just outside the drop-off zone while another delivers. Emergent patience.

---

## Try it yourself

**Experiment 1 — Competitive rewards.**
Give each robot a reward only for packages *it* delivers, not total throughput. Watch behaviour shift: robots start "stealing" packages near other robots' pickup zones. Cooperation collapses. Same algorithm, different reward signal, completely different society.

**Experiment 2 — Reduce visibility.**
Cut each robot's observation radius from 5 to 2 cells. Agents make worse decisions near edges and corners — they can't see far enough to plan. But watch how they compensate: they become more conservative, staying near areas they know well.

**Experiment 3 — Add a fifth robot.**
More robots = more throughput, up to a point. Beyond that, the additional robot adds more congestion than it removes. Find the point where adding robots starts hurting performance.

---

## Next

IPPO produces cooperative behaviour without any explicit coordination mechanism. But what about emergent behaviours that are far more complex — collective intelligence that arises purely from agents chasing simple individual goals?
