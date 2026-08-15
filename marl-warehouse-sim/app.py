"""
MARL Warehouse Coordinator
A full multi-agent reinforcement learning warehouse simulation:
  — Custom grid warehouse (shelves, pickup zones, delivery zones)
  — IPPO-trained robots that learn to deliver packages and avoid collisions
  — Strategy comparison: trained vs random vs greedy
  — Live training with reward/delivery/collision analytics
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from agents.ippo import (
    CHECKPOINT,
    TrainingState,
    WarehouseIPPO,
    start_training,
)
from viz.charts import comparison_chart, training_dashboard
from warehouse.env import N_ACTIONS, WarehouseEnv
from warehouse.renderer import frames_to_gif, render_frame

# ── Global state ──────────────────────────────────────────────────────────────
_train_state = TrainingState()
_ippo: WarehouseIPPO | None = None


def _get_ippo(n_robots: int) -> WarehouseIPPO | None:
    global _ippo
    env_probe = WarehouseEnv(n_robots=n_robots)
    ippo = WarehouseIPPO(n_robots, env_probe.obs_dim)
    if ippo.load(CHECKPOINT):
        _ippo = ippo
        return ippo
    return None


def _run_episode(
    n_robots: int,
    policy: str,  # "trained" | "random" | "greedy" | "noop"
    max_steps: int = 100,
    seed: int = 0,
    fps: int = 10,
) -> tuple[str | None, dict]:
    env = WarehouseEnv(n_robots=n_robots, max_steps=max_steps, seed=seed)
    obs = env.reset(seed=seed)

    ippo = None
    if policy == "trained":
        ippo = _get_ippo(n_robots)
        if ippo is None:
            return None, {
                "deliveries": 0,
                "collisions": 0,
                "reward": 0.0,
                "error": "no_model",
            }

    frames = []
    trails: list[list[tuple[int, int]]] = [[] for _ in range(n_robots)]
    total_rewards = 0.0

    for step in range(max_steps):
        # Render current state
        for i, r in enumerate(env.robots):
            trails[i].append((r.row, r.col))
            if len(trails[i]) > 12:
                trails[i].pop(0)

        frame = render_frame(
            env.grid,
            env.robots,
            trails=trails,
            step=step,
            deliveries=env.total_deliveries,
            collisions=env.total_collisions,
            title=policy.upper(),
        )
        frames.append(frame)

        # Choose actions
        if policy == "trained" and ippo:
            actions_dict = ippo.greedy_all(obs)
            actions = [actions_dict[aid] for aid in env.agent_ids]
        elif policy == "greedy":
            # Greedy: move toward current goal
            actions = []
            for robot in env.robots:
                dr = robot.goal_row - robot.row
                dc = robot.goal_col - robot.col
                if abs(dr) >= abs(dc):
                    actions.append(1 if dr > 0 else 2)
                elif dc != 0:
                    actions.append(4 if dc > 0 else 3)
                else:
                    actions.append(0)
        elif policy == "noop":
            actions = [0] * n_robots
        else:  # random
            actions = [int(np.random.randint(N_ACTIONS)) for _ in range(n_robots)]

        obs, rewards, done, _info = env.step(actions)
        total_rewards += sum(rewards)
        if done:
            break

    gif_path = frames_to_gif(frames, fps=fps)
    return gif_path, {
        "deliveries": env.total_deliveries,
        "collisions": env.total_collisions,
        "reward": total_rewards,
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────


def cb_run_sim(n_robots, policy_label, fps, progress: gr.Progress = gr.Progress()):
    policy_map = {
        "🤖 Trained (IPPO)": "trained",
        "🎲 Random": "random",
        "➡️ Greedy": "greedy",
    }
    policy = policy_map[policy_label]
    progress(0.1, desc="Initialising warehouse…")
    gif, stats = _run_episode(
        int(n_robots), policy, max_steps=100, seed=42, fps=int(fps)
    )
    progress(0.9, desc="Building GIF…")

    if gif is None and stats.get("error") == "no_model":
        status_md = (
            "❌ **No trained model found.** Train agents first in the **Train** tab."
        )
        return None, status_md

    status_md = f"""
### Simulation Complete

| Metric | Value |
|---|---|
| **Strategy** | {policy_label} |
| **Robots** | {n_robots} |
| **Deliveries** | `{stats["deliveries"]}` |
| **Collisions** | `{stats["collisions"]}` |
| **Total Reward** | `{stats["reward"]:+.2f}` |
"""
    progress(1.0)
    return gif, status_md


def cb_start_train(n_robots, total_eps, rollout, lr):
    global _train_state
    if _train_state.running:
        return "⚠️ Training already running."
    _train_state = TrainingState()
    start_training(
        n_robots=int(n_robots),
        total_episodes=int(total_eps),
        rollout_steps=int(rollout),
        state=_train_state,
        lr=float(lr),
    )
    return f"✅ Training started — {int(n_robots)} robots, {int(total_eps)} episodes."


def cb_stop_train():
    _train_state.running = False
    return "⏹ Stop requested."


def cb_refresh_train():
    fig = training_dashboard(_train_state)
    return fig, _train_state.status


def cb_benchmark(n_robots, progress: gr.Progress = gr.Progress()):
    results = {}
    n = int(n_robots)
    seed = 99

    progress(0.1, desc="Running IPPO…")
    _, stats = _run_episode(n, "trained", seed=seed)
    if stats.get("error"):
        results["IPPO (trained)"] = {
            "deliveries": 0,
            "collisions": 0,
            "reward": 0.0,
            "note": "⚠️ Not trained",
        }
    else:
        results["IPPO (trained)"] = stats

    progress(0.4, desc="Running Random…")
    _, stats = _run_episode(n, "random", seed=seed)
    results["Random"] = stats

    progress(0.65, desc="Running Greedy…")
    _, stats = _run_episode(n, "greedy", seed=seed)
    results["Greedy"] = stats

    progress(0.85, desc="Building chart…")
    fig = comparison_chart(results)

    rows = "\n".join(
        f"| {name} | `{d['deliveries']}` | `{d['collisions']}` | `{d['reward']:+.2f}` |"
        for name, d in results.items()
    )
    summary = f"""
### Benchmark: {n} robots, 100 steps

| Strategy | Deliveries | Collisions | Reward |
|---|---|---|---|
{rows}

> **IPPO** should deliver more packages with fewer collisions than random.
> **Greedy** picks the next best move each step — no coordination.
"""
    progress(1.0)
    return fig, summary


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0d1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1150px !important; margin: 0 auto !important; }

.wh-header {
    border-bottom: 1px solid #21262d;
    padding: 1.4rem 1.5rem 0.9rem;
}
.wh-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    font-weight: 700; color: #1890ff; margin: 0;
}
.wh-sub { color: #4a5568; font-size: 0.85rem; margin-top: 0.3rem; }
.wh-badges { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.7rem; }
.wh-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    letter-spacing:0.08em; padding:3px 10px; border-radius:3px;
    text-transform:uppercase;
}
.b-blue   { background:#0a1a30; color:#1890ff; border:1px solid #1a3a6e; }
.b-green  { background:#061a0e; color:#10b981; border:1px solid #0a4a26; }
.b-amber  { background:#1e1206; color:#f59e0b; border:1px solid #5c3a06; }
.b-red    { background:#1a0808; color:#ef4444; border:1px solid #5c1a1a; }

.tab-nav { border-bottom:1px solid #21262d !important; background:transparent !important; }
.tab-nav button {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.72rem !important; letter-spacing:0.05em !important;
    color:#4a5568 !important; background:transparent !important;
    border:none !important; padding:0.7rem 1.1rem !important;
    text-transform:uppercase !important;
}
.tab-nav button.selected {
    color:#1890ff !important; border-bottom:2px solid #1890ff !important;
}

button.primary {
    font-family:'JetBrains Mono',monospace !important; font-weight:600 !important;
    background:linear-gradient(135deg,#0a2a52,#0d3a72) !important;
    color:#58a6ff !important; border:1px solid #1890ff !important;
    border-radius:5px !important; transition:all 0.2s !important;
}
button.primary:hover { box-shadow:0 0 12px rgba(24,144,255,0.3) !important; }
button.secondary {
    font-family:'JetBrains Mono',monospace !important;
    background:#161b22 !important; color:#10b981 !important;
    border:1px solid #10b981 !important; border-radius:5px !important;
}
button.stop {
    background:#180a0a !important; color:#ef4444 !important;
    border:1px solid #ef4444 !important; border-radius:5px !important;
}

label span, .gradio-container label {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.7rem !important; color:#4a5568 !important;
    text-transform:uppercase !important; letter-spacing:0.06em !important;
}
input[type=range] { -webkit-appearance:none; height:3px;
                    background:#21262d; border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:#1890ff; cursor:pointer;
    border:2px solid #0d1117;
}
textarea, .gradio-container textarea {
    font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
    background:#060a12 !important; color:#1890ff !important;
    border:1px solid #21262d !important; border-radius:4px !important;
}
.gradio-container h2 { color:#1890ff !important; font-family:'JetBrains Mono',monospace !important; }
.gradio-container h3 { color:#10b981 !important; }
.gradio-container p  { color:#64748b !important; }
table { width:100%; border-collapse:collapse; }
th { background:#161b22; color:#1890ff; font-family:'JetBrains Mono',monospace;
     font-size:0.7rem; text-align:left; padding:7px 12px;
     border-bottom:1px solid #21262d; text-transform:uppercase; }
td { padding:7px 12px; border-bottom:1px solid #0d1117;
     color:#e2e8f0; font-size:0.83rem; }
code { font-family:'JetBrains Mono',monospace; background:#161b22;
       color:#f59e0b; padding:1px 5px; border-radius:3px; }
blockquote { border-left:3px solid #1890ff; padding:0.6rem 1rem;
             background:#161b22; border-radius:0 4px 4px 0; margin:0.5rem 0; }
footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MARL Warehouse Coordinator") as demo:
    gr.HTML("""
    <div class="wh-header">
        <div class="wh-title">📦 MARL Warehouse Coordinator</div>
        <div class="wh-sub">
            Multi-Agent Reinforcement Learning · Custom Grid Warehouse ·
            IPPO · Task-Based Coordination
        </div>
        <div class="wh-badges">
            <span class="wh-badge b-blue">● Custom Grid Env</span>
            <span class="wh-badge b-green">● IPPO Training</span>
            <span class="wh-badge b-amber">● Package Delivery</span>
            <span class="wh-badge b-red">● Collision Avoidance</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Live Simulation
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🏭 SIMULATION"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    LIVE WAREHOUSE SIMULATION
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Watch robots navigate the warehouse, pick up packages from
                    <strong style="color:#2ea043">P zones</strong> and deliver them to
                    <strong style="color:#1890ff">D zones</strong>.
                    Coloured trails show recent paths.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    sim_robots = gr.Slider(
                        2, 6, value=4, step=1, label="Number of robots"
                    )
                    sim_policy = gr.Radio(
                        ["🤖 Trained (IPPO)", "🎲 Random", "➡️ Greedy"],
                        value="🎲 Random",
                        label="Robot strategy",
                    )
                    sim_fps = gr.Slider(
                        5, 20, value=10, step=1, label="Animation speed (fps)"
                    )
                    btn_sim = gr.Button("▶ RUN SIMULATION", variant="primary")

                    gr.HTML("""
                    <div style="background:#161b22;border:1px solid #21262d;
                                border-radius:6px;padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#4a5568;text-transform:uppercase;margin-bottom:0.5rem;">
                            LEGEND
                        </div>
                        <div style="font-size:0.82rem;color:#64748b;line-height:1.9;">
                            <div><span style="color:#2ea043">■</span> <strong style="color:#e2e8f0">P</strong> — Pickup zone (load package)</div>
                            <div><span style="color:#1890ff">■</span> <strong style="color:#e2e8f0">D</strong> — Delivery zone (unload package)</div>
                            <div><span style="color:#e2e8f0">●</span> — Robot (number = ID)</div>
                            <div><span style="color:#e2e8f0">□</span> — Package carried</div>
                            <div>Dashed line = current task route</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=1):
                    sim_gif = gr.Image(
                        label="Warehouse Replay", type="filepath", height=380
                    )
                    sim_status = gr.Markdown("*Click Run Simulation to start.*")

            btn_sim.click(
                cb_run_sim, [sim_robots, sim_policy, sim_fps], [sim_gif, sim_status]
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Train Agents
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🧠 TRAIN AGENTS"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    IPPO TRAINING PIPELINE
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Independent PPO — each robot learns its own navigation policy.
                    Reward signal: +2 per delivery, −0.3 per collision, −0.01 per step.
                    Trained model is saved and auto-loaded in the Simulation tab.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    train_robots = gr.Slider(
                        2, 6, value=4, step=1, label="Number of robots"
                    )
                    train_eps = gr.Slider(
                        50, 1000, value=200, step=50, label="Training episodes"
                    )
                    train_rollout = gr.Slider(
                        50, 200, value=100, step=50, label="Steps per episode"
                    )
                    train_lr = gr.Slider(
                        1e-5, 1e-3, value=3e-4, step=1e-5, label="Learning rate"
                    )
                    with gr.Row():
                        btn_train = gr.Button("▶ START TRAINING", variant="primary")
                        btn_stop = gr.Button("⏹ STOP", variant="stop")
                    btn_refresh = gr.Button("🔄 REFRESH METRICS", variant="secondary")
                    train_msg = gr.Textbox(label="Status", lines=2, interactive=False)

                    gr.HTML("""
                    <div style="background:#161b22;border:1px solid #21262d;border-radius:6px;
                                padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#4a5568;text-transform:uppercase;margin-bottom:0.6rem;">
                            REWARD STRUCTURE
                        </div>
                        <div style="display:grid;grid-template-columns:auto 1fr;gap:0.3rem 0.8rem;">
                            <div style="color:#10b981;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">+2.0</div>
                            <div style="color:#64748b;font-size:0.8rem;">Package delivered</div>
                            <div style="color:#10b981;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">+0.5</div>
                            <div style="color:#64748b;font-size:0.8rem;">Package picked up</div>
                            <div style="color:#ef4444;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">−0.3</div>
                            <div style="color:#64748b;font-size:0.8rem;">Robot collision</div>
                            <div style="color:#ef4444;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">−0.01</div>
                            <div style="color:#64748b;font-size:0.8rem;">Per step (efficiency)</div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    train_status_md = gr.Markdown(
                        "*Start training to see live metrics.*"
                    )
                    train_fig = gr.Plot(label="Training Dashboard")

            btn_train.click(
                cb_start_train,
                [train_robots, train_eps, train_rollout, train_lr],
                [train_msg],
            )
            btn_stop.click(cb_stop_train, outputs=[train_msg])
            btn_refresh.click(cb_refresh_train, outputs=[train_fig, train_status_md])

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Benchmark
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🏁 BENCHMARK"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">
                    STRATEGY COMPARISON
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    All three strategies on the same warehouse, same seed.
                    Measures deliveries, collisions, and total reward.
                </div>
            </div>
            """)

            with gr.Row():
                bench_robots = gr.Slider(
                    2, 6, value=4, step=1, label="Number of robots"
                )
                btn_bench = gr.Button("🏁 RUN BENCHMARK", variant="primary")

            bench_fig = gr.Plot(label="Strategy Comparison")
            bench_summary = gr.Markdown(
                "*Click Run Benchmark to compare all strategies.*"
            )

            btn_bench.click(cb_benchmark, [bench_robots], [bench_fig, bench_summary])

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — How It Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 HOW IT WORKS"):
            gr.Markdown("""
## The Warehouse Problem

N robots operate in a 12×12 grid warehouse with:
- **Shelf zones** (dark blue) — impassable structures
- **Pickup zones** `P` — robots collect packages here
- **Delivery zones** `D` — robots drop off packages here
- **Open corridors** — navigable floor space

Each robot has a task: go to a pickup zone, collect a package, deliver it.
When a delivery is complete, a new task is immediately assigned.

The challenge: robots must coordinate to avoid collisions while maximising
the number of deliveries per episode.

---

## Independent PPO (IPPO)

Each robot runs its own PPO loop — no shared weights, no centralised critic.

**Why IPPO works here:**
- Shared reward structure (all robots benefit from efficient navigation)
- Simple enough that independent learning converges
- Scales to any number of agents

**Observation per robot** (13-dimensional):
```
[own_row, own_col,           # where am I?
 goal_row, goal_col,         # where am I going?
 has_package,                # am I carrying something?
 rel_robot_1_r, rel_1_c,    # where are nearby robots?
 rel_robot_2_r, rel_2_c,
 rel_robot_3_r, rel_3_c,
 rel_robot_4_r, rel_4_c]
```

**Action space:** 5 discrete — stay, up, down, left, right

**Reward:**
```
+2.0 — successful delivery
+0.5 — package picked up
−0.3 — collision with another robot
−0.01 — per step (encourages efficiency)
```

---

## Strategy Comparison

| Strategy | How it works | Coordination |
|---|---|---|
| **IPPO (trained)** | Learned policy from thousands of episodes | Implicit (emergent) |
| **Greedy** | Always move toward current goal | None — ignores others |
| **Random** | Random action each step | None |

The benchmark shows how much coordination is learned: IPPO should have
more deliveries and fewer collisions than greedy or random.

---

## Grid Environment Design

The warehouse layout is fixed across all episodes. Key design choices:
- **Bottleneck corridors** force robots to coordinate to avoid deadlocks
- **Multiple pickup/delivery zones** allow parallel task execution
- **Collision penalty** discourages robots from occupying the same cell

When two robots want the same cell at the same step, **both are blocked** —
this simulates physical collision without complex physics.
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                color:#21262d;padding:1.5rem 0 0.5rem;border-top:1px solid #161b22;
                letter-spacing:0.1em;text-transform:uppercase;">
        MARL Warehouse · IPPO · Custom Grid Env · PyTorch · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
