"""
Green Logistics Optimizer — DQN Emission Reduction Platform
End-to-end urban delivery route optimization using Deep Q-Networks.

Compares three strategies on the same city scenario:
  · DQN Agent      — learned policy (trains to minimise carbon cost)
  · Greedy Heuristic — always moves toward goal (baseline)
  · A* Optimal     — shortest path, true distance lower bound
"""

from __future__ import annotations

import gradio as gr
from core.agents import (
    RouteResult,
    TrainingState,
    run_astar,
    run_dqn,
    run_greedy,
    start_training,
)
from core.env import PRESETS, GreenCityEnv, parse_congestion
from viz.charts import (
    carbon_trace,
    city_map,
    empty_fig,
    performance_radar,
    training_curve,
)

# ── Shared training state ─────────────────────────────────────────────────────
_train_state = TrainingState()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_env(
    size, start_y, start_x, goal_y, goal_x, congestion_text, vehicle
) -> GreenCityEnv:
    size = int(size)
    zones = parse_congestion(congestion_text, size)
    env = GreenCityEnv(size=size)
    env.reset(
        start_pos=[int(start_y), int(start_x)],
        goal_pos=[int(goal_y), int(goal_x)],
        congestion_map=zones,
        vehicle=vehicle,
    )
    return env


def _summary_html(results: list[RouteResult], vehicle: str) -> str:
    rows = ""
    for r in results:
        col = {
            "DQN Agent": "#00e676",
            "Greedy": "#f44336",
            "A* Optimal": "#29b6f6",
        }.get(r.agent_name.split("(")[0].strip(), "#e6edf3")
        rows += f"""
        <tr>
          <td style="color:{col};font-weight:600">{r.agent_name}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r.total_carbon:.3f} kg</td>
          <td style="font-family:'JetBrains Mono',monospace">{r.steps}</td>
          <td style="font-family:'JetBrains Mono',monospace">{r.congestion_hits}</td>
          <td style="color:{"#00e676" if r.delivered else "#f44336"}">
            {"✅ Delivered" if r.delivered else "❌ Max steps"}
          </td>
        </tr>"""

    # Carbon saving of DQN vs Greedy
    dqn_r = next((r for r in results if "DQN" in r.agent_name), None)
    greedy_r = next((r for r in results if "Greedy" in r.agent_name), None)
    saving_html = ""
    if dqn_r and greedy_r and greedy_r.total_carbon > 0:
        saved_pct = (
            (greedy_r.total_carbon - dqn_r.total_carbon) / greedy_r.total_carbon * 100
        )
        col = "#00e676" if saved_pct > 0 else "#f44336"
        saving_html = f"""
        <div style="margin-top:1rem;padding:0.8rem 1rem;background:rgba(0,230,118,0.08);
                    border:1px solid rgba(0,230,118,0.2);border-radius:8px;font-family:'JetBrains Mono',monospace;">
          <span style="color:#8b949e;font-size:0.8rem;">DQN carbon saving vs Greedy: </span>
          <strong style="color:{col};font-size:1.1rem;">{saved_pct:+.1f}%</strong>
          &nbsp;·&nbsp;Vehicle: <strong style="color:#e6edf3">{vehicle}</strong>
        </div>"""

    return f"""
<div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
    <thead>
      <tr style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                 text-transform:uppercase;color:#8b949e;border-bottom:1px solid #21262d;">
        <th style="padding:8px 12px;text-align:left">Strategy</th>
        <th style="padding:8px 12px;text-align:left">Carbon</th>
        <th style="padding:8px 12px;text-align:left">Steps</th>
        <th style="padding:8px 12px;text-align:left">Congestion Hits</th>
        <th style="padding:8px 12px;text-align:left">Outcome</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {saving_html}
</div>"""


# ── Callbacks ─────────────────────────────────────────────────────────────────


def cb_load_preset(preset_name: str):
    p = PRESETS[preset_name]
    cong_str = "; ".join(f"{z[0]} {z[1]}" for z in p["congestion"])
    return (
        p["size"],
        p["start"][0],
        p["start"][1],
        p["goal"][0],
        p["goal"][1],
        cong_str,
        f"*{p['description']}*",
    )


def cb_run_mission(
    size,
    start_y,
    start_x,
    goal_y,
    goal_x,
    congestion_text,
    vehicle,
    run_dqn_flag,
    run_greedy_flag,
    run_astar_flag,
    progress: gr.Progress = gr.Progress(),
):
    try:
        progress(0.05, desc="Building city environment…")
        env = _make_env(
            size, start_y, start_x, goal_y, goal_x, congestion_text, vehicle
        )

        results: list[RouteResult] = []

        if run_dqn_flag:
            progress(0.2, desc="Running DQN agent…")
            env2 = _make_env(
                size, start_y, start_x, goal_y, goal_x, congestion_text, vehicle
            )
            results.append(run_dqn(env2))

        if run_greedy_flag:
            progress(0.5, desc="Running Greedy heuristic…")
            env3 = _make_env(
                size, start_y, start_x, goal_y, goal_x, congestion_text, vehicle
            )
            results.append(run_greedy(env3))

        if run_astar_flag:
            progress(0.7, desc="Running A* optimal…")
            env4 = _make_env(
                size, start_y, start_x, goal_y, goal_x, congestion_text, vehicle
            )
            results.append(run_astar(env4))

        if not results:
            return (
                empty_fig("Select at least one strategy."),
                empty_fig(""),
                empty_fig(""),
                "*No strategies selected.*",
            )

        progress(0.85, desc="Building charts…")
        map_fig = city_map(env, results)
        trace_fig = carbon_trace(results)
        radar_fig = performance_radar(results, int(size))
        summary = _summary_html(results, vehicle)

        progress(1.0)
        return map_fig, trace_fig, radar_fig, summary

    except Exception as e:
        err = empty_fig(f"Error: {e}")
        return err, err, err, f"❌ **Error:** {e}"


def cb_start_train(total_steps: int):
    global _train_state
    if _train_state.running:
        return "⚠️ Training already running."
    _train_state = TrainingState()
    start_training(int(total_steps), _train_state)
    return f"✅ DQN training started — {int(total_steps):,} steps."


def cb_stop_train():
    _train_state.running = False
    return "⏹ Stop requested."


def cb_refresh_train():
    return training_curve(_train_state), _train_state.status


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:     #0a0b10;
    --card:   rgba(17, 19, 30, 0.85);
    --border: rgba(255,255,255,0.07);
    --accent: #00e676;
    --text:   #e6edf3;
    --dim:    #8b949e;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

.glo-header {
    text-align: center; padding: 1.8rem 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
}
.glo-title {
    font-size: clamp(1.5rem, 4vw, 2.3rem); font-weight: 700;
    background: linear-gradient(135deg, #00e676, #1de9b6, #00bcd4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem; letter-spacing: -0.01em;
}
.glo-sub { color: var(--dim); font-size: 0.88rem; }
.glo-badges { display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap; margin-top:0.7rem; }
.g-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    letter-spacing:0.08em; padding:3px 10px; border-radius:20px;
    text-transform:uppercase;
}
.b-green  { background:rgba(0,230,118,0.1); color:#00e676; border:1px solid rgba(0,230,118,0.25); }
.b-red    { background:rgba(244,67,54,0.1);  color:#f44336; border:1px solid rgba(244,67,54,0.25); }
.b-blue   { background:rgba(41,182,246,0.1); color:#29b6f6; border:1px solid rgba(41,182,246,0.25); }
.b-amber  { background:rgba(255,109,0,0.1);  color:#ff6d00; border:1px solid rgba(255,109,0,0.25); }

.tab-nav { border-bottom:1px solid var(--border) !important; background:transparent !important; }
.tab-nav button {
    font-family:'Outfit',sans-serif !important; font-size:0.82rem !important;
    font-weight:500 !important; color:var(--dim) !important;
    background:transparent !important; border:none !important;
    padding:0.65rem 1.1rem !important;
}
.tab-nav button.selected { color:#00e676 !important; border-bottom:2px solid #00e676 !important; }

.glass-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

button.primary {
    font-family:'Outfit',sans-serif !important; font-weight:600 !important;
    background:linear-gradient(135deg,#00a854,#00e676) !important;
    color:#000 !important; border:none !important;
    border-radius:8px !important; transition:all 0.2s !important;
}
button.primary:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }
button.secondary {
    font-family:'Outfit',sans-serif !important;
    background:rgba(0,230,118,0.08) !important; color:#00e676 !important;
    border:1px solid rgba(0,230,118,0.2) !important; border-radius:8px !important;
}
button.stop {
    background:rgba(244,67,54,0.08) !important; color:#f44336 !important;
    border:1px solid rgba(244,67,54,0.2) !important; border-radius:8px !important;
    font-family:'Outfit',sans-serif !important;
}

label span, .gradio-container label {
    font-family:'JetBrains Mono',monospace !important; font-size:0.7rem !important;
    color:var(--dim) !important; text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}
input[type=range] { -webkit-appearance:none; height:3px;
                    background:rgba(255,255,255,0.08); border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:#00e676; cursor:pointer;
    border:2px solid var(--bg);
}
textarea, .gradio-container textarea {
    font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
    background:rgba(255,255,255,0.03) !important; color:#00e676 !important;
    border:1px solid var(--border) !important; border-radius:6px !important;
}
.gradio-container h2 { color:#00e676 !important; font-family:'JetBrains Mono',monospace !important; }
.gradio-container h3 { color:#1de9b6 !important; }
.gradio-container p  { color:var(--dim) !important; }
table { width:100%; border-collapse:collapse; }
th { background:#111318; color:#00e676; font-family:'JetBrains Mono',monospace;
     font-size:0.7rem; text-align:left; padding:7px 12px;
     border-bottom:1px solid var(--border); text-transform:uppercase; }
td { padding:7px 12px; border-bottom:1px solid rgba(255,255,255,0.03);
     color:var(--text); font-size:0.85rem; }
code { font-family:'JetBrains Mono',monospace; background:rgba(0,230,118,0.12);
       color:#00e676; padding:1px 5px; border-radius:3px; }
blockquote { border-left:3px solid #00e676; padding:0.5rem 1rem;
             background:rgba(0,230,118,0.06); border-radius:0 6px 6px 0; margin:0.5rem 0; }
footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Green Logistics Optimizer") as demo:
    gr.HTML("""
    <div class="glo-header">
        <div class="glo-title">🌿 Green Logistics Optimizer</div>
        <div class="glo-sub">
            Deep Q-Network · Urban Delivery Route Optimization ·
            Carbon Emission Minimization
        </div>
        <div class="glo-badges">
            <span class="g-badge b-green">● DQN Agent</span>
            <span class="g-badge b-red">● Greedy Baseline</span>
            <span class="g-badge b-blue">● A* Optimal</span>
            <span class="g-badge b-amber">● Congestion Zones</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Mission Control
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🗺️ MISSION CONTROL"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;">
                    CONFIGURE CITY & DEPLOY FLEET
                </div>
                <div style="color:#8b949e;font-size:0.85rem;margin-top:0.2rem;">
                    Select a scenario preset or configure manually.
                    Run all three strategies and compare their carbon footprints.
                </div>
            </div>
            """)

            with gr.Row():
                # ── Config panel ──────────────────────────────────────────
                with gr.Column(scale=1, min_width=300, elem_classes="glass-card"):
                    gr.HTML(
                        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#8b949e;text-transform:uppercase;margin-bottom:0.5rem;\">SCENARIO PRESET</div>"
                    )
                    preset_dd = gr.Dropdown(
                        list(PRESETS.keys()),
                        value=next(iter(PRESETS.keys())),
                        label="Load Preset",
                    )
                    preset_desc = gr.Markdown("*Select a preset to load.*")
                    btn_load = gr.Button("📋 Load Preset", variant="secondary")

                    gr.HTML(
                        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#8b949e;text-transform:uppercase;margin:0.8rem 0 0.4rem;\">CITY CONFIG</div>"
                    )
                    grid_size = gr.Slider(
                        4, 12, value=7, step=1, label="Grid size (N×N)"
                    )
                    with gr.Row():
                        start_y = gr.Number(
                            value=0, label="Start row", precision=0, minimum=0
                        )
                        start_x = gr.Number(
                            value=0, label="Start col", precision=0, minimum=0
                        )
                    with gr.Row():
                        goal_y = gr.Number(
                            value=6, label="Goal row", precision=0, minimum=0
                        )
                        goal_x = gr.Number(
                            value=6, label="Goal col", precision=0, minimum=0
                        )

                    vehicle = gr.Dropdown(
                        ["Diesel", "EV"],
                        value="Diesel",
                        label="Vehicle type",
                        info="EV: 0.2× base carbon · Diesel: 1.0×",
                    )

                    cong_text = gr.Textbox(
                        value="2 2; 2 3; 3 2; 4 4; 4 5",
                        label="Congestion zones  (y x; y x; ...)",
                        info="Enter row col pairs, semicolon-separated",
                        lines=2,
                    )

                    gr.HTML(
                        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#8b949e;text-transform:uppercase;margin:0.8rem 0 0.4rem;\">STRATEGIES</div>"
                    )
                    run_dqn_cb = gr.Checkbox(label="DQN Agent", value=True)
                    run_greedy_cb = gr.Checkbox(label="Greedy Heuristic", value=True)
                    run_astar_cb = gr.Checkbox(label="A* Optimal", value=True)

                    btn_run = gr.Button("🚀 DEPLOY FLEET", variant="primary")

                # ── Output panel ──────────────────────────────────────────
                with gr.Column(scale=2):
                    summary_html = gr.HTML("*Deploy the fleet to see results.*")
                    city_img = gr.Image(
                        label="City Carbon Map",
                        type="pil",
                        show_label=False,
                        height=480,
                    )

            # Preset loader
            btn_load.click(
                cb_load_preset,
                [preset_dd],
                [grid_size, start_y, start_x, goal_y, goal_x, cong_text, preset_desc],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Analytics
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 ANALYTICS"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;">
                    ROUTE ANALYTICS & PERFORMANCE RADAR
                </div>
            </div>
            """)

            with gr.Row():
                trace_img = gr.Image(
                    label="Carbon Trace", type="pil", show_label=False, height=320
                )
                radar_img = gr.Image(
                    label="Performance Radar", type="pil", show_label=False, height=320
                )

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Training Lab
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("⚗️ TRAINING LAB"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;">
                    TRAIN DQN AGENT FROM SCRATCH
                </div>
                <div style="color:#8b949e;font-size:0.85rem;margin-top:0.2rem;">
                    Trains on a 7×7 city with random congestion. The saved model is
                    used automatically when you run DQN in Mission Control.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass-card"):
                    t_steps = gr.Slider(
                        5_000,
                        100_000,
                        value=20_000,
                        step=5_000,
                        label="Training timesteps",
                        info="~20k = fast (~30s) · ~50k = better policy",
                    )
                    with gr.Row():
                        btn_train = gr.Button("▶ START TRAINING", variant="primary")
                        btn_stop = gr.Button("⏹ STOP", variant="stop")
                    btn_refresh = gr.Button("🔄 REFRESH", variant="secondary")
                    t_msg = gr.Textbox(label="Status", lines=2, interactive=False)

                    gr.HTML("""
                    <div style="background:rgba(0,230,118,0.07);border:1px solid rgba(0,230,118,0.15);
                                border-radius:8px;padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#00e676;text-transform:uppercase;margin-bottom:0.5rem;">
                            DQN CONFIG
                        </div>
                        <div style="font-size:0.8rem;color:#8b949e;line-height:1.9;">
                            <div>Policy: <span style="color:#e6edf3">MlpPolicy</span></div>
                            <div>Learning rate: <span style="color:#e6edf3">1×10⁻³</span></div>
                            <div>Buffer: <span style="color:#e6edf3">50,000</span></div>
                            <div>Batch size: <span style="color:#e6edf3">64</span></div>
                            <div>Gamma: <span style="color:#e6edf3">0.95</span></div>
                            <div>Exploration: <span style="color:#e6edf3">30% → 5%</span></div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    t_status = gr.Markdown("*Start training to see live metrics.*")
                    t_chart = gr.Image(
                        label="Training", type="pil", show_label=False, height=300
                    )

            btn_train.click(cb_start_train, [t_steps], [t_msg])
            btn_stop.click(cb_stop_train, outputs=[t_msg])
            btn_refresh.click(cb_refresh_train, outputs=[t_chart, t_status])

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — How DQN Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 HOW DQN WORKS"):
            gr.Markdown("""
## Deep Q-Network (DQN) for Route Optimization

DQN is an off-policy reinforcement learning algorithm that learns a value function
$Q(s, a)$ — the expected discounted reward of taking action $a$ in state $s$.

---

## The Logistics Environment

**State:** `[agent_y, agent_x]` — 2-D position on the delivery grid

**Actions:** 4 discrete — Up, Down, Left, Right

**Reward per step:**
```
base_cost   = 1.0 (Diesel)  or  0.2 (EV)
multiplier  = 4.0 if in congestion zone,  else 1.0
step_reward = -(base_cost × multiplier)
delivery    = +20.0 when goal reached
```

The agent learns to find routes that minimise total carbon cost while still
reaching the delivery destination.

---

## The Q-Learning Update (Bellman Equation)

At each step the network is trained to satisfy:

$$Q(s, a) = r + \\gamma \\max_{a'} Q(s', a')$$

**DQN key innovations** over vanilla Q-learning:

| Innovation | Benefit |
|---|---|
| **Experience Replay** | Stores transitions in a buffer; samples random mini-batches to break correlations |
| **Target Network** | A frozen copy of Q updated periodically — stabilises training |
| **ε-greedy Exploration** | ε decays from 30% → 5% — explores early, exploits later |

---

## Three Strategies Compared

| Strategy | Algorithm | Knows future? | Carbon optimal? |
|---|---|---|---|
| **DQN Agent** | Learned neural Q-function | No (learned from experience) | Learns to avoid congestion |
| **Greedy** | Move toward goal each step | No | Ignores congestion |
| **A* Optimal** | Shortest path search | Yes (has full map) | Minimises distance, not carbon |

> A* finds the shortest path but **ignores emission cost** — it may still
> cross congestion zones if that's the direct route. DQN learns to avoid them.

---

## Carbon Cost Formula

$$C_{step} = \\text{base\\_cost} \\times \\begin{cases} 4.0 & \\text{if congestion zone} \\\\ 1.0 & \\text{otherwise} \\end{cases}$$

$$C_{total} = \\sum_{t=1}^{T} C_{step}^{(t)}$$

**EV advantage:** base cost = 0.2 (5× lower than Diesel).
Even in congestion (0.2 × 4 = 0.8), EV beats Diesel on clear roads (1.0).

---

## Reading the Charts

- **City map**: darker red = higher carbon zones. Green path = DQN, Red = Greedy, Blue = A*.
- **Carbon trace**: cumulative cost — lower final value = better
- **Performance radar**: 5 metrics normalised to 0-10:
  - **Eco Score**: inverse of total carbon
  - **Speed**: fewer steps = higher score
  - **Safety**: fewer congestion crossings = higher score
  - **Efficiency**: distance covered per step
  - **Delivery**: binary — 10 if delivered, 2 if not
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                color:#21262d;padding:1.5rem 0 0.5rem;border-top:1px solid rgba(255,255,255,0.05);
                letter-spacing:0.1em;text-transform:uppercase;margin-top:1rem;">
        DQN · A* · Greedy · Stable-Baselines3 · Gymnasium · Gradio
    </div>
    """)

    # ── Mission Control wiring ─────────────────────────────────────────────────
    btn_run.click(
        cb_run_mission,
        inputs=[
            grid_size,
            start_y,
            start_x,
            goal_y,
            goal_x,
            cong_text,
            vehicle,
            run_dqn_cb,
            run_greedy_cb,
            run_astar_cb,
        ],
        outputs=[city_img, trace_img, radar_img, summary_html],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
