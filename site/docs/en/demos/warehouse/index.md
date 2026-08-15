---
description: "Warehouse Robots demo — 4 IPPO agents coordinate on a 12×12 grid to deliver packages without collisions. Watch emergent traffic rules develop without any programming."
---

# 🤖 Warehouse Robots

Four agents operate in a 12×12 warehouse grid. Each agent's job: pick up packages and deliver them to drop-off zones. No agent is told about the others. No coordination protocol. No central controller.

Yet by episode 1,000, they've developed informal traffic lanes.

<a href="https://huggingface.co/spaces/Dash10107/marl-warehouse-sim" target="_blank" rel="noopener" class="hero-btn-primary" style="display:inline-flex;margin-bottom:1.5rem">
  ▶ Open Live Demo ↗
</a>

---

## What this demo shows

**The phase transition.** This is the thing to watch. For the first 200–400 episodes, agents crash into each other constantly. The grid is chaotic. Then — usually between episode 300 and 500 — something shifts. Agents start avoiding each other. Deliveries per episode jumps.

This isn't programmed. It emerges because collisions reduce individual reward. Avoiding others is individually rational, and individual rationality becomes collective coordination.

**The traffic patterns.** By episode 1,000+, open the "paths" overlay. You'll see informal lanes — agents moving in one direction tend to cluster on one side of the grid. Nobody planned this. It's a Nash equilibrium: any agent that switches sides reduces its own efficiency.

**Individual vs joint reward.** Toggle between "individual" and "joint" reward views. Notice that individual reward maximisation produces a collectively good outcome — but only because the reward function was designed to make it so. Change the reward to penalise others' success and watch cooperation collapse instantly.

**Policy heterogeneity.** Each agent runs an independent PPO policy. Over time, agents *specialise* — some tend to pick up packages in the top half, others in the bottom. This isn't assigned. It's a stable equilibrium that reduces collision probability.

---

## Try these experiments

**Experiment 1 — 2 agents vs 4 agents vs 8 agents.**
Run each configuration. With 2 agents, coordination is easy and emerges fast. With 8, the grid gets crowded — coordination takes longer and the traffic patterns are messier. Notice how sample efficiency decreases with agent count.

**Experiment 2 — Competitive reward.**
Switch to competitive mode (each agent is penalised for others' deliveries). Watch what happens: agents start blocking each other's paths. The same algorithm, the same environment — opposite emergent behaviour. Incentive design is everything.

**Experiment 3 — One agent goes rogue.**
Freeze three agents' policies after episode 800. Let only one continue training. Watch it adapt to the now-static environment. It finds optimal routes around the frozen agents within 200 episodes. This is the power of online learning.

---

## The architecture

Each agent is an independent PPO actor with:
- **Local observation:** own position, nearby agent positions, nearby package locations (within radius 3)
- **Action space:** 5 discrete (up, down, left, right, pick up/drop)
- **Shared policy architecture** but individual parameter updates
- **No communication channel:** agents cannot send messages to each other

This is the **IPPO** (Independent PPO) architecture — the simplest and most common MARL baseline, and often competitive with more complex methods.

---

## The chapters behind this demo

- **[What Happens With Two of You?](../../modules/multi-agent-rl/)** — IPPO, Nash equilibrium, CTDE, and QMIX
- **[When No One Planned It](../../modules/swarm/)** — emergent cooperation and the mathematics of emergence

---

**Difficulty:** Advanced · **Algorithm:** IPPO · **Agents:** 4 · **Action space:** Discrete
