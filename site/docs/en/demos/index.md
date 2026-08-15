---
description: "Interactive reinforcement learning demos — run Q-Learning, DQN, PPO, SAC, and RLHF agents in your browser. No installation required."
---

# Live Demos

Every algorithm in this course has a live demo you can run right now in your browser. No Python. No installation. Just open and watch.

Each demo is paired with a chapter — the chapter explains the theory, the demo shows it moving.

<div class="card-grid">
  <a href="https://huggingface.co/spaces/Dash10107/rl_maze_solver" class="card" target="_blank" rel="noopener">
    <h3>🧩 Maze Solver</h3>
    <p>Q-Learning, SARSA, and Monte Carlo race through procedurally generated mazes. Watch the Q-value heatmap fill in as the agent learns.</p>
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem">
      <span class="card-tag">Beginner</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">Q-Learning</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">SARSA</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">Monte Carlo</span>
    </div>
  </a>
  <a href="https://huggingface.co/spaces/Dash10107/rocket-lander-sac" class="card" target="_blank" rel="noopener">
    <h3>🚀 Rocket Lander</h3>
    <p>SAC controls continuous engine thrust to land a rocket with sub-pixel precision. Entropy regularisation keeps the policy adaptive under wind perturbation.</p>
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem">
      <span class="card-tag">Advanced</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">SAC</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">Continuous Control</span>
    </div>
  </a>
  <a href="https://huggingface.co/spaces/Dash10107/marl-warehouse-sim" class="card" target="_blank" rel="noopener">
    <h3>🤖 Warehouse Robots</h3>
    <p>4 agents coordinate on a 12×12 grid to pick and deliver packages without colliding. Informal traffic rules emerge from individual reward signals.</p>
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem">
      <span class="card-tag">Advanced</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">IPPO</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">Multi-Agent</span>
    </div>
  </a>
  <a href="https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf" class="card" target="_blank" rel="noopener">
    <h3>✍️ Calligrapher (RLHF)</h3>
    <p>Label pairs of brush strokes as "better" or "worse." A reward model learns your taste. PPO trains the agent to produce strokes you'd approve of.</p>
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem">
      <span class="card-tag">Advanced</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">RLHF</span>
      <span class="card-tag" style="background:var(--vp-c-bg-soft)">PPO</span>
    </div>
  </a>
</div>

---

## What to look for in each demo

The demos are most useful if you know what to watch. Here's a guide for each:

### Maze Solver — what to watch

**The Q-value heatmap.** Start with an empty maze. As the agent learns, the heatmap fills in — bright colours near the exit, darker further away. This is the Bellman equation propagating backward from the reward.

**Compare algorithms.** Switch between Q-Learning, SARSA, and Monte Carlo. Notice:
- Q-Learning fills the heatmap faster
- SARSA stays further from walls (safer)
- Monte Carlo has blank runs early (needs full episodes to learn)

**The policy path.** Late in training, switch to "policy view." You'll see arrows in each cell pointing toward the exit. That's the greedy policy — the learned solution.

→ Read the chapter: [Teaching an Agent to Remember](../modules/q-learning/)

---

### Rocket Lander — what to watch

**The action distribution.** On the right panel, watch the thrust output distribution. Early training: wide bell curve (high entropy, lots of exploration). Late training: narrow spike (the agent is confident in its thrust profile).

**Turbulence test.** Enable wind. The SAC-trained agent adapts — it has retained some randomness, so small perturbations don't break it. Compare to a fully deterministic policy: it breaks immediately.

**The entropy curve.** Watch `α × H(π)` over training. It doesn't go to zero — it stabilises at a healthy minimum. The agent stays slightly unpredictable by design.

→ Read the chapter: [Continuous Control (SAC)](../modules/sac/)

---

### Warehouse Robots — what to watch

**Episode 1.** Four robots crash into each other constantly. No coordination. The grid is chaos.

**The phase transition.** Around episode 200–400, something shifts. Robots start avoiding each other. Not because they were told to — because collisions reduce individual reward. Coordination emerges from individual optimisation.

**The traffic pattern.** By episode 1000+, informal lanes appear. Robots going left cluster on one side. Going right, on the other. Nobody programmed this. Watch for it.

→ Read the chapter: [What Happens With Two of You?](../modules/multi-agent-rl/)

---

### Calligrapher RLHF — what to watch

**Pre-training strokes.** Before any labels, the strokes are random scribbles. The policy is uniform — it has no preference signal.

**After 20 labels.** The reward model has enough signal to distinguish good from bad. Strokes start drifting toward what you preferred. They're not perfect, but the direction is right.

**Try disagreeing with yourself.** Label the first 10 pairs preferring thin strokes. Label the next 10 preferring thick strokes. Watch the reward model become confused — its loss plateaus. Consistency of preference matters.

→ Read the chapter: [Teaching AI What You Actually Want (RLHF)](../modules/rlhf/)

---

## Can't access a demo?

The demos run on Hugging Face Spaces. If a Space is sleeping (cold start), it may take 30–60 seconds to wake up on first load. If it's been idle for several days, it may need to be restarted — open an issue on [GitHub](https://github.com/Dash10107/reinforcement-learning-lab) and it'll be back up within a day.
