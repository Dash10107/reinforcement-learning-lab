"""
AI Tutor Pro — A2C Personalized Learning Path Optimizer
End-to-end reinforcement learning platform for adaptive education:
  · Actor-Critic (A2C) recommends which subject to study next
  · Live simulation shows 10–50 step learning trajectory
  · Policy probability charts reveal how the agent thinks
  · Training lab lets you retrain the agent from scratch
  · Analytics: attention allocation, reward curve, convergence
"""

from __future__ import annotations
import time
import numpy as np
import gradio as gr

from core.environment import SUBJECTS, SUBJECT_COLORS, N_SUBJECTS
from core.agent import (
    TrainingState, load_model, get_policy_probs,
    simulate_path, start_training, MODEL_PATH,
)
from viz.charts import (
    trajectory_chart, policy_bars, episode_analytics,
    training_chart, _empty,
)

# ── Load model on startup ─────────────────────────────────────────────────────
_model = load_model(MODEL_PATH)
_train_state = TrainingState()

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _stat_val(val: str, color: str = "#f8fafc") -> str:
    return f"<div class='stat-val' style='color:{color}'>{val}</div>"


def _prob_bars_html(probs: list[float]) -> str:
    best = int(np.argmax(probs))
    html = ""
    for i, (name, prob) in enumerate(zip(SUBJECTS, probs)):
        w   = prob * 100
        col = SUBJECT_COLORS[i]
        crown = " 👑" if i == best else ""
        html += f"""
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;
                      font-size:0.78rem;color:#94a3b8;margin-bottom:4px">
            <span style="color:{col if i==best else '#94a3b8'};
                         font-weight:{'600' if i==best else '400'}">
              {name}{crown}
            </span>
            <span style="font-family:'JetBrains Mono',monospace">{w:.1f}%</span>
          </div>
          <div style="height:6px;background:rgba(255,255,255,0.05);
                      border-radius:3px;overflow:hidden">
            <div style="height:100%;width:{w}%;background:{col};
                        transition:width 0.5s ease;opacity:{'1' if i==best else '0.55'}">
            </div>
          </div>
        </div>"""
    return html


def _insights_html(probs: list[float], avg: float, votes: int = 0) -> str:
    best   = int(np.argmax(probs))
    conf   = max(probs) * 100
    second = sorted(range(len(probs)), key=lambda i: -probs[i])[1]
    return f"""
<div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">
  <div style="margin-bottom:8px">
    The A2C policy assigns <strong style="color:{SUBJECT_COLORS[best]}">
    {SUBJECTS[best]}</strong> the highest probability at
    <strong style="color:#f8fafc">{conf:.1f}%</strong>.
  </div>
  <div style="margin-bottom:8px">
    Second choice: <strong style="color:{SUBJECT_COLORS[second]}">
    {SUBJECTS[second]}</strong> ({probs[second]*100:.1f}%).
  </div>
  <div>
    Current average proficiency: <strong style="color:#f8fafc">
    {avg:.1f}%</strong>.
  </div>
</div>"""


# ── Callbacks ─────────────────────────────────────────────────────────────────

def cb_analyze(*vals):
    pct  = list(vals)
    avg  = sum(pct) / len(pct)
    probs = get_policy_probs(_model, pct).tolist()
    best  = int(np.argmax(probs))
    conf  = max(probs) * 100

    chart = policy_bars(probs, pct)

    return (
        _stat_val(f"{avg:.1f}%"),
        _stat_val(f"{conf:.1f}%", "#6366f1"),
        _stat_val(SUBJECTS[best], SUBJECT_COLORS[best]),
        _prob_bars_html(probs),
        _insights_html(probs, avg),
        chart,
    )


def cb_simulate(*vals_and_steps):
    *pct_vals, n_steps = vals_and_steps
    pct    = list(pct_vals)
    n_steps = int(n_steps)

    history = simulate_path(_model, pct, n_steps=n_steps, deterministic=True)

    for i, step_data in enumerate(history):
        state_pct = step_data["state"]
        probs     = step_data["probs"]
        action    = step_data["action"]
        avg       = sum(state_pct) / len(state_pct)
        conf      = max(probs) * 100
        best      = int(np.argmax(probs))

        status_html = f"""
<div style="display:flex;align-items:center;gap:12px;padding:8px 14px;
            background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);
            border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#a5b4fc">
  <span>STEP {step_data['step']}/{n_steps}</span>
  <span style="color:#f8fafc">→</span>
  <span style="color:{SUBJECT_COLORS[action]}">{SUBJECTS[action]}</span>
  <span style="color:#64748b">|</span>
  <span>reward: {step_data['reward']:.3f}</span>
  {'<span style="color:#10b981">✓ MASTERED</span>' if step_data["done"] else ""}
</div>"""

        yield (
            *state_pct,                              # 5 sliders
            _stat_val(f"{avg:.1f}%"),               # avg
            _stat_val(f"{conf:.1f}%", "#6366f1"),   # conf
            _stat_val(SUBJECTS[best], SUBJECT_COLORS[best]),  # focus
            _prob_bars_html(probs),                  # bars
            _insights_html(probs, avg),              # insights
            status_html,                             # step status
        )
        time.sleep(0.35)
        if step_data["done"]:
            break

    # Final charts (after loop finishes)
    traj   = trajectory_chart(history)
    epan   = episode_analytics(history)
    # Clear status
    yield (
        *history[-1]["state"],
        _stat_val(f"{sum(history[-1]['state'])/len(history[-1]['state']):.1f}%"),
        _stat_val(f"{max(history[-1]['probs'])*100:.1f}%", "#6366f1"),
        _stat_val(SUBJECTS[int(np.argmax(history[-1]['probs']))],
                  SUBJECT_COLORS[int(np.argmax(history[-1]['probs']))]),
        _prob_bars_html(history[-1]["probs"]),
        _insights_html(history[-1]["probs"],
                       sum(history[-1]["state"])/len(history[-1]["state"])),
        "<div></div>",   # clear status
    )


def cb_start_training(total_steps: int):
    global _train_state
    if _train_state.running:
        return "⚠️ Training already running.", gr.update()
    _train_state = TrainingState()
    start_training(int(total_steps), _train_state)
    return f"✅ Training started — {int(total_steps):,} steps.", gr.update()


def cb_stop_training():
    _train_state.running = False
    return "⏹ Stop requested."


def cb_refresh_training():
    global _model
    fig = training_chart(_train_state)
    if _train_state.model_ready:
        try:
            _model = load_model(MODEL_PATH)
            note = "  ✓ Model reloaded."
        except Exception:
            note = ""
    else:
        note = ""
    return fig, _train_state.status + note


def cb_get_traj_charts(*vals_and_steps):
    *pct_vals, n_steps = vals_and_steps
    history = simulate_path(_model, list(pct_vals), n_steps=int(n_steps))
    return trajectory_chart(history), episode_analytics(history)


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:     #0a0b10;
    --card:   rgba(17, 19, 24, 0.85);
    --border: rgba(255,255,255,0.07);
    --accent: #6366f1;
    --text:   #f8fafc;
    --dim:    #64748b;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* Header */
.tutor-header {
    text-align: center; padding: 2rem 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
}
.tutor-title {
    font-size: clamp(1.5rem, 4vw, 2.4rem); font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.01em; margin: 0 0 0.3rem;
}
.tutor-sub { color: var(--dim); font-size: 0.9rem; }
.tutor-badges { display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap; margin-top:0.8rem; }
.t-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:0.08em;
    padding:3px 10px; border-radius:20px; text-transform:uppercase;
    background:rgba(99,102,241,0.1); color:#818cf8;
    border:1px solid rgba(99,102,241,0.25);
}

/* Glass cards */
.glass-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* Tabs */
.tab-nav { border-bottom:1px solid var(--border) !important; background:transparent !important; }
.tab-nav button {
    font-family:'Outfit',sans-serif !important; font-size:0.82rem !important;
    font-weight:500 !important; color:var(--dim) !important;
    background:transparent !important; border:none !important;
    padding:0.65rem 1.1rem !important;
}
.tab-nav button.selected { color:#818cf8 !important; border-bottom:2px solid #6366f1 !important; }

/* Stat cards */
.stat-header { font-size:0.7rem; color:var(--dim); text-transform:uppercase;
               letter-spacing:1px; margin-bottom:6px; font-family:'JetBrains Mono',monospace; }
.stat-val { font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; }

/* Buttons */
button.primary {
    font-family:'Outfit',sans-serif !important; font-weight:600 !important;
    background:linear-gradient(135deg,#4f46e5,#6366f1) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; transition:all 0.2s !important;
}
button.primary:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }
button.secondary {
    font-family:'Outfit',sans-serif !important;
    background:rgba(99,102,241,0.1) !important; color:#818cf8 !important;
    border:1px solid rgba(99,102,241,0.3) !important; border-radius:10px !important;
}
button.stop {
    background:rgba(239,68,68,0.1) !important; color:#f87171 !important;
    border:1px solid rgba(239,68,68,0.3) !important; border-radius:10px !important;
    font-family:'Outfit',sans-serif !important;
}

/* Sliders */
label span, .gradio-container label {
    font-family:'Outfit',sans-serif !important; font-size:0.82rem !important;
    color:var(--dim) !important;
}
input[type=range] { -webkit-appearance:none; height:4px;
                    background:rgba(255,255,255,0.08); border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:16px; height:16px;
    border-radius:50%; background:var(--accent); cursor:pointer;
    border:2px solid var(--bg);
}

/* Textareas */
textarea, .gradio-container textarea {
    font-family:'JetBrains Mono',monospace !important; font-size:0.8rem !important;
    background:rgba(255,255,255,0.04) !important; color:#818cf8 !important;
    border:1px solid var(--border) !important; border-radius:8px !important;
}

/* Markdown */
.gradio-container h2 { color:#818cf8 !important; font-size:1.1rem !important; }
.gradio-container h3 { color:#a5b4fc !important; }
.gradio-container p  { color:var(--dim) !important; }
table { width:100%; border-collapse:collapse; }
th { background:#111318; color:#818cf8; font-family:'JetBrains Mono',monospace;
     font-size:0.7rem; text-align:left; padding:7px 12px;
     border-bottom:1px solid var(--border); text-transform:uppercase; }
td { padding:7px 12px; border-bottom:1px solid rgba(255,255,255,0.04);
     color:var(--text); font-size:0.85rem; }
code { font-family:'JetBrains Mono',monospace; background:rgba(99,102,241,0.15);
       color:#a5b4fc; padding:1px 5px; border-radius:3px; }

footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── Chart.js radar (live slider update) ───────────────────────────────────────

RADAR_JS = """
(s0, s1, s2, s3, s4) => {
    const vals  = [s0, s1, s2, s3, s4];
    const cols  = ['#6366f1','#10b981','#f59e0b','#ec4899','#3b82f6'];
    if (!window.rc) {
        const el = document.getElementById('tutor-radar');
        if (!el) return;
        window.rc = new Chart(el.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Math','Physics','Lit','History','CS'],
                datasets: [{
                    data: vals,
                    backgroundColor: 'rgba(99,102,241,0.12)',
                    borderColor: '#6366f1', borderWidth: 2.5,
                    pointBackgroundColor: cols,
                    pointBorderColor: '#0a0b10', pointRadius: 5,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { r: {
                    min: 0, max: 100, angleLines: { color:'rgba(255,255,255,0.06)' },
                    grid: { color:'rgba(255,255,255,0.06)' }, ticks: { display:false },
                    pointLabels: { color:'#94a3b8', font:{ size:11 } }
                }},
                plugins: { legend: { display:false } }
            }
        });
    } else {
        window.rc.data.datasets[0].data = vals;
        window.rc.update('none');
    }
}
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Tutor Pro — A2C Learning Path Optimizer") as demo:

    gr.HTML("""
    <div class="tutor-header">
        <div class="tutor-title">AI Tutor Pro</div>
        <div class="tutor-sub">
            Actor-Critic (A2C) Personalized Learning Path Optimizer
        </div>
        <div class="tutor-badges">
            <span class="t-badge">A2C Policy</span>
            <span class="t-badge">5 Subjects</span>
            <span class="t-badge">Real-Time Simulation</span>
            <span class="t-badge">Training Lab</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Dashboard
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Dashboard"):

            with gr.Row(equal_height=False):

                # ── Sidebar: sliders ──────────────────────────────────────
                with gr.Column(scale=1, elem_classes="glass-card", min_width=260):
                    gr.HTML("""
                    <div style="margin-bottom:1rem;">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                    letter-spacing:1px;font-family:'JetBrains Mono',monospace;">
                            Student Proficiency
                        </div>
                        <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.3rem;">
                            Drag sliders to set current knowledge levels, then
                            analyse or simulate the optimal learning path.
                        </div>
                    </div>
                    """)

                    s_math = gr.Slider(0, 100, value=25, step=1, label="Mathematics")
                    s_phys = gr.Slider(0, 100, value=30, step=1, label="Physics")
                    s_lit  = gr.Slider(0, 100, value=40, step=1, label="Literature")
                    s_hist = gr.Slider(0, 100, value=20, step=1, label="History")
                    s_cs   = gr.Slider(0, 100, value=35, step=1, label="Computer Science")
                    s_list = [s_math, s_phys, s_lit, s_hist, s_cs]

                    gr.HTML("<div style='height:0.5rem'></div>")
                    btn_analyze  = gr.Button("🔍 Analyse State", variant="primary")
                    btn_simulate = gr.Button("▶ Simulate Path", variant="secondary")

                # ── Main panel ────────────────────────────────────────────
                with gr.Column(scale=3):

                    # Stat cards row
                    with gr.Row():
                        with gr.Column(elem_classes="glass-card"):
                            gr.HTML("<div class='stat-header'>Average Proficiency</div>")
                            v_avg = gr.HTML(_stat_val("—"))
                        with gr.Column(elem_classes="glass-card"):
                            gr.HTML("<div class='stat-header'>Policy Confidence</div>")
                            v_conf = gr.HTML(_stat_val("—", "#6366f1"))
                        with gr.Column(elem_classes="glass-card"):
                            gr.HTML("<div class='stat-header'>Recommended Focus</div>")
                            v_focus = gr.HTML(_stat_val("—", "#6366f1"))

                    # Radar + Insights row
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes="glass-card"):
                            gr.HTML("""
                            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                        letter-spacing:1px;font-family:'JetBrains Mono',monospace;
                                        margin-bottom:0.8rem;">Live Radar</div>
                            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                            <div style="position:relative;height:320px;">
                                <canvas id="tutor-radar"></canvas>
                            </div>
                            """)
                            sim_status = gr.HTML("<div></div>")

                        with gr.Column(scale=2, elem_classes="glass-card"):
                            gr.HTML("<div class='stat-header'>Action Probabilities</div>")
                            v_bars = gr.HTML(
                                "<div style='color:#64748b;font-size:0.82rem;font-style:italic;'>"
                                "Click Analyse to see policy probabilities.</div>"
                            )
                            gr.HTML("<div style='height:1px;background:rgba(255,255,255,0.06);margin:12px 0'></div>")
                            gr.HTML("<div class='stat-header'>Agent Insights</div>")
                            v_insights = gr.HTML(
                                "<div style='color:#64748b;font-size:0.82rem;font-style:italic;'>"
                                "Proficiency analysis pending.</div>"
                            )

            # Policy chart (below main grid)
            with gr.Row():
                policy_chart = gr.Image(label="Policy Analysis Chart",
                                        show_label=False, type="pil", height=280)

            # Simulation config
            with gr.Row():
                n_steps_slider = gr.Slider(5, 50, value=20, step=5,
                                           label="Simulation steps")

            # Wire up
            for s in s_list:
                s.change(None, inputs=s_list, outputs=None, js=RADAR_JS)

            btn_analyze.click(
                cb_analyze, inputs=s_list,
                outputs=[v_avg, v_conf, v_focus, v_bars, v_insights, policy_chart],
            )

            btn_simulate.click(
                cb_simulate,
                inputs=[*s_list, n_steps_slider],
                outputs=[*s_list, v_avg, v_conf, v_focus, v_bars, v_insights, sim_status],
            )

            demo.load(None, inputs=s_list, outputs=None, js=RADAR_JS)

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Trajectory & Analytics
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📈 Analytics"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">
                    LEARNING TRAJECTORY ANALYSIS
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Run a simulation then view the full trajectory chart,
                    agent attention allocation, and cumulative reward.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=260, elem_classes="glass-card"):
                    an_math = gr.Slider(0, 100, value=25, step=1, label="Mathematics")
                    an_phys = gr.Slider(0, 100, value=30, step=1, label="Physics")
                    an_lit  = gr.Slider(0, 100, value=40, step=1, label="Literature")
                    an_hist = gr.Slider(0, 100, value=20, step=1, label="History")
                    an_cs   = gr.Slider(0, 100, value=35, step=1, label="Computer Science")
                    an_steps = gr.Slider(5, 50, value=25, step=5, label="Steps")
                    btn_an  = gr.Button("📈 Generate Analytics", variant="primary")

                with gr.Column(scale=3):
                    an_traj = gr.Image(label="Trajectory", show_label=False,
                                       type="pil", height=380)
                    an_ep   = gr.Image(label="Episode Analytics", show_label=False,
                                       type="pil", height=250)

            btn_an.click(
                cb_get_traj_charts,
                inputs=[an_math, an_phys, an_lit, an_hist, an_cs, an_steps],
                outputs=[an_traj, an_ep],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Training Lab
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("⚗️ Training Lab"):
            gr.HTML("""
            <div style="padding:0.7rem 0 0.3rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                            color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">
                    RETRAIN THE A2C AGENT FROM SCRATCH
                </div>
                <div style="color:#64748b;font-size:0.85rem;margin-top:0.2rem;">
                    Train a new policy using different timestep budgets.
                    The retrained model auto-loads after training completes.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass-card"):
                    t_steps = gr.Slider(5_000, 100_000, value=20_000, step=5_000,
                                        label="Training timesteps")
                    with gr.Row():
                        btn_train   = gr.Button("▶ Start Training", variant="primary")
                        btn_stop_t  = gr.Button("⏹ Stop",  variant="stop")
                    btn_refresh = gr.Button("🔄 Refresh", variant="secondary")
                    t_msg       = gr.Textbox(label="Status", lines=2, interactive=False)

                    gr.HTML("""
                    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                                border-radius:8px;padding:0.9rem;margin-top:0.8rem;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                    color:#6366f1;text-transform:uppercase;margin-bottom:0.5rem;">
                            A2C Hyperparameters
                        </div>
                        <div style="font-size:0.8rem;color:#64748b;line-height:1.9;">
                            <div>Policy: <span style="color:#a5b4fc">MlpPolicy (64×64)</span></div>
                            <div>Learning rate: <span style="color:#a5b4fc">7×10⁻⁴</span></div>
                            <div>Discount γ: <span style="color:#a5b4fc">0.99</span></div>
                            <div>n_steps: <span style="color:#a5b4fc">5</span></div>
                            <div>Entropy coef: <span style="color:#a5b4fc">0.01</span></div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=2):
                    t_status_md = gr.Markdown("*Start training to see live metrics.*")
                    t_chart     = gr.Image(label="Training Chart", show_label=False,
                                           type="pil", height=300)

            btn_train.click(cb_start_training, [t_steps], [t_msg, gr.State()])
            btn_stop_t.click(cb_stop_training, outputs=[t_msg])
            btn_refresh.click(cb_refresh_training, outputs=[t_chart, t_status_md])

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — How A2C Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 How A2C Works"):
            gr.Markdown("""
## Actor-Critic (A2C) — The Algorithm

A2C is an on-policy reinforcement learning algorithm that simultaneously
maintains two neural networks:

| Network | Input | Output | Role |
|---|---|---|---|
| **Actor** `π_θ(a|s)` | Student state | Action probabilities | Decides which subject to study |
| **Critic** `V_φ(s)` | Student state | State value | Estimates expected future reward |

---

## The Tutoring Environment

**State:** Proficiency scores $s = [p_1, p_2, p_3, p_4, p_5] \in [0, 1]^5$ — one per subject.

**Action:** Which subject to focus on: $a \in \{0, 1, 2, 3, 4\}$

**Transition dynamics at each step:**
```
p_a  ← min(1.0,  p_a + Uniform(0.12, 0.28))   # studying boosts focus subject
p_i  ← max(0.0,  p_i - Uniform(0.005, 0.025))  # forgetting reduces all others
```

**Reward:** Current proficiency of the chosen subject — encourages the agent
to focus on subjects where it can make concrete progress.

**Terminal condition:** All $p_i \geq 0.98$ (mastery across all subjects)

---

## The A2C Update

At each step the advantage is computed:

$$A(s, a) = r + \gamma V_\phi(s') - V_\phi(s)$$

**Actor loss** (maximise expected advantage):
$$\mathcal{L}_\pi = -\log \pi_\theta(a|s) \cdot A(s,a) - \beta H(\pi_\theta(\cdot|s))$$

The entropy term $H$ (weight $\beta=0.01$) encourages exploration.

**Critic loss** (minimise Bellman residual):
$$\mathcal{L}_V = (r + \gamma V_\phi(s') - V_\phi(s))^2$$

---

## Why A2C for Tutoring?

| Property | Benefit |
|---|---|
| **On-policy** | Directly optimises the current policy — no stale experience |
| **Advantage** | Reduces variance vs pure policy gradient |
| **Discrete actions** | Naturally fits "choose a subject" decisions |
| **Fast convergence** | 10k–50k steps is enough for this 5-dim environment |

The agent learns that the optimal strategy is not to always study the weakest
subject — it considers which subject provides the best reward given current
proficiency and the forgetting dynamics of all other subjects.

---

## Reading the Dashboard

- **Policy Confidence**: `max(π(a|s))` — how decisively the agent recommends one subject
- **Action Probabilities**: full distribution over all 5 subjects
- **Simulation**: deterministic rollout (`argmax`) — shows the greedy policy path
- **Trajectory chart**: proficiency per subject over 20 steps — should all converge to 98%
- **Attention bar**: which subject the agent focused on at each step
""")

    gr.HTML("""
    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#1e2a3d;padding:1.5rem 0 0.5rem;border-top:1px solid rgba(255,255,255,0.06);
                letter-spacing:0.1em;text-transform:uppercase;margin-top:1rem;">
        A2C Policy · Stable-Baselines3 · Gymnasium · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
