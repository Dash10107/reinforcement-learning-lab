"""
The Swarm Architect — MARL Experience
An end-to-end interactive app for anyone to understand, watch, and train
multi-agent AI systems — no ML background required.
"""

from __future__ import annotations

import os
import threading

import gradio as gr
import numpy as np
from agents.ippo import IPPOTrainer
from config import ENV, NET, PPO, TRAIN
from environment.wrapper import get_agent_ids, make_env
from evaluation.metrics import TrainingLog, evaluate_agents
from visualization.animator import (
    frames_to_gif,
    make_reward_history_plot,
    make_training_plot,
)

# ── Global state ──────────────────────────────────────────────────────────────
_trainer: IPPOTrainer | None = None
_agent_ids: list[str] = []
_log = TrainingLog()
_training_active = False
_training_thread: threading.Thread | None = None
_status_message = "Ready."


def _init_trainer() -> IPPOTrainer:
    global _trainer, _agent_ids  # noqa: PLW0602
    probe = make_env(ENV, render=False)
    _agent_ids = get_agent_ids(probe)
    probe.close()
    t = IPPOTrainer(ENV, PPO, NET)
    t.init_agents(_agent_ids)
    return t


def _load_pretrained(trainer: IPPOTrainer) -> bool:
    for p in [os.path.join(TRAIN.checkpoint_dir, "best_agent.pt"), "shared_actor.pth"]:
        if os.path.exists(p):
            try:
                trainer.load_best(p)
                return True
            except Exception:  # noqa: BLE001, S110
                pass
    return False


def _reward_to_score(reward: float) -> tuple[str, str, str]:
    """Convert raw team reward to human-readable grade, label, and colour."""
    per_agent = reward / max(ENV.n_agents, 1)
    if per_agent >= -1.5:
        return "A", "🏆 Excellent — agents are perfectly coordinated!", "#2ddb7c"
    if per_agent >= -2.5:
        return "B", "✅ Good — most landmarks are covered", "#4fb3ff"
    if per_agent >= -4.0:
        return "C", "⚠️ Learning — agents spreading but still colliding", "#f5a623"
    return "F", "💥 Early stage — agents still figuring things out", "#ff4d6d"


def _training_encouragement(rolling: float, episode: int) -> str:
    """Return a plain-English message based on current training progress."""
    per_agent = rolling / max(ENV.n_agents, 1)
    if episode < 20:
        return "🔍 Agents are exploring randomly — this is normal at the start."
    if per_agent < -4.0:
        return "🌀 Agents are still bumping into each other — give them more time."
    if per_agent < -2.5:
        return "📡 Progress! Agents are starting to spread out."
    if per_agent < -1.5:
        return "🚀 Great coordination forming — agents are learning to cover landmarks."
    return "🏆 Agents are working together beautifully!"


# ── Training loop ─────────────────────────────────────────────────────────────
def _train_loop(episodes: int, lr: float, gamma: float, clip: float, rollout: int):
    global _trainer, _log, _training_active, _status_message  # noqa: PLW0602
    PPO.lr_actor = lr
    PPO.gamma = gamma
    PPO.clip_epsilon = clip
    PPO.rollout_steps = rollout

    env = make_env(ENV, render=False)
    best_rolling = float("-inf")

    for ep in range(1, episodes + 1):
        if not _training_active:
            break
        rewards, metrics = _trainer.train_episode(env)
        _log.record(ep, rewards, metrics)
        if _log.rolling_mean > best_rolling:
            best_rolling = _log.rolling_mean
            os.makedirs(TRAIN.checkpoint_dir, exist_ok=True)
            _trainer.save_all(TRAIN.checkpoint_dir, 0)
        encouragement = _training_encouragement(_log.rolling_mean, ep)
        _status_message = (
            f"{encouragement}\n"
            f"Episode {ep}/{episodes}  ·  "
            f"Team score: {_log.mean_rewards[-1]:+.2f}  ·  "
            f"Best so far: {best_rolling:+.2f}"
        )

    env.close()
    _training_active = False
    grade, label, _ = _reward_to_score(best_rolling)
    _status_message = f"Training complete! Grade: {grade} — {label}"


# ── Callbacks ─────────────────────────────────────────────────────────────────


def cb_watch_demo(steps: int, progress: gr.Progress = gr.Progress()):  # noqa: B008
    """Load the pretrained model and render an episode — the main 'Watch' action."""
    global _trainer
    progress(0.1, desc="Loading agents…")
    if _trainer is None:
        _trainer = _init_trainer()
    _load_pretrained(_trainer)

    env = make_env(ENV, render=True)
    obs_dict, _ = env.reset()
    frames, total_rewards = [], {aid: 0.0 for aid in _trainer.agents}

    progress(0.3, desc="Running episode…")
    for s in range(int(steps)):
        actions = {
            aid: _trainer.agents[aid].greedy_action(
                obs_dict.get(aid, np.zeros(ENV.obs_dim))
            )
            for aid in _trainer.agents
        }
        obs_dict, rewards, terms, truncs, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        for aid in total_rewards:
            total_rewards[aid] += rewards.get(aid, 0.0)
        if all(terms.get(a, False) or truncs.get(a, False) for a in _trainer.agents):
            break

    env.close()
    progress(0.9, desc="Building replay…")

    if not frames:
        return None, "❌ Could not render frames.", ""

    gif_path = frames_to_gif(frames, fps=12)
    team_r = sum(total_rewards.values())
    grade, label, _ = _reward_to_score(team_r)

    score_md = f"""
### Episode Result

| | |
|---|---|
| **Coordination Grade** | **{grade}** |
| **Verdict** | {label} |
| **Team Score** | `{team_r:+.2f}` |
| **Steps Taken** | `{len(frames)}` |
| **Agents** | `{len(_trainer.agents)}` |

> **How to read the score:** The environment gives negative rewards.
> A score near **0** means perfect coverage with no collisions.
> A score below **−10** means agents are still bumping into each other.
"""
    progress(1.0, desc="Done!")
    return gif_path, score_md, grade


def cb_quick_train():
    """One-click training with sensible defaults."""
    global _trainer, _log, _training_active, _training_thread
    if _training_active:
        return (
            "⚠️ Training is already running. Check the Mission Report tab.",
            gr.update(),
        )

    _trainer = _init_trainer()
    _load_pretrained(_trainer)
    _log = TrainingLog()
    _training_active = True

    _training_thread = threading.Thread(
        target=_train_loop,
        args=(200, PPO.lr_actor, PPO.gamma, PPO.clip_epsilon, PPO.rollout_steps),
        daemon=True,
    )
    _training_thread.start()
    return (
        "✅ Quick training started! 200 episodes with recommended settings.\n"  # noqa: ISC004
        "Go to the **📈 Mission Report** tab and click **Refresh** to watch progress.",
        gr.update(),
    )


def cb_custom_train(episodes, lr, gamma, clip, rollout):
    global _trainer, _log, _training_active, _training_thread
    if _training_active:
        return "⚠️ Training is already running.", gr.update()
    _trainer = _init_trainer()
    _load_pretrained(_trainer)
    _log = TrainingLog()
    _training_active = True
    _training_thread = threading.Thread(
        target=_train_loop,
        args=(int(episodes), float(lr), float(gamma), float(clip), int(rollout)),
        daemon=True,
    )
    _training_thread.start()
    return f"✅ Custom training started for {int(episodes)} episodes.", gr.update()


def cb_stop_training():
    global _training_active
    _training_active = False
    return "⏹ Stop requested — finishing current episode…"


def cb_refresh_report():
    data = _log.to_dict()
    fig1 = make_training_plot(data)
    fig2 = make_reward_history_plot(data)

    n = len(_log.episodes)
    if n == 0:
        summary = "*No training data yet. Start training first.*"
        return _status_message, fig1, fig2, summary

    grade, label, _ = _reward_to_score(_log.rolling_mean * ENV.n_agents)
    summary = f"""
### 📊 Training Summary

| Metric | Value |
|---|---|
| **Episodes completed** | `{n}` |
| **Current grade** | **{grade}** — {label} |
| **Best team score** | `{_log.best_reward * ENV.n_agents:+.2f}` |
| **Recent average** | `{_log.rolling_mean * ENV.n_agents:+.2f}` (last 50 episodes) |

**What's a good score?**
`0` = perfect. Agents cover all 5 landmarks with zero collisions.
`−5` = decent. Minor gaps and occasional bumps.
`−15+` = still learning. Keep training!
"""
    return _status_message, fig1, fig2, summary


def cb_evaluate(n_eps: int):
    global _trainer
    if _trainer is None:
        _trainer = _init_trainer()
        if not _load_pretrained(_trainer):
            return "❌ No trained model found. Train your swarm first in the **Train** tab."

    env = make_env(ENV, render=False)
    results = evaluate_agents(env, _trainer.agents, n_episodes=int(n_eps))
    env.close()

    team_r = results["team_reward"]
    grade, label, _ = _reward_to_score(team_r)
    per_agent_md = "\n".join(
        f"| Agent {i + 1} | `{r:+.3f}` |"
        for i, (_, r) in enumerate(results["mean_reward_per_agent"].items())
    )

    return f"""
### Evaluation over {n_eps} episode(s)

**Overall Grade: {grade}** — {label}

| Metric | Value |
|---|---|
| **Team Score** | `{team_r:+.3f}` |
| **Avg Episode Length** | `{results["mean_episode_length"]:.0f} steps` |

**Per-Agent Breakdown:**

| Agent | Score |
|---|---|
{per_agent_md}

> Scores are negative because the task penalises uncovered landmarks.
> The closer to **0**, the better the team is coordinating.
"""


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* Hero */
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 700;
    background: linear-gradient(135deg, #4ECDC4, #45B7D1, #96CEB4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
}
.hero-sub {
    color: #64748b; font-size: 1rem; letter-spacing: 0.02em;
}

/* Tabs */
.tab-nav { border-bottom: 1px solid #1e293b !important; background: transparent !important; }
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    color: #64748b !important; background: transparent !important;
    border: none !important; padding: 0.75rem 1.25rem !important;
    transition: color 0.2s !important;
}
.tab-nav button.selected { color: #4ECDC4 !important; border-bottom: 2px solid #4ECDC4 !important; }
.tab-nav button:hover { color: #94a3b8 !important; }

/* Cards */
.info-card {
    background: #111827; border: 1px solid #1e293b; border-radius: 12px;
    padding: 1.25rem; height: 100%;
}
.info-card-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.info-card-title { font-weight: 600; font-size: 1rem; color: #e2e8f0; margin-bottom: 0.4rem; }
.info-card-body { color: #94a3b8; font-size: 0.88rem; line-height: 1.6; }

/* Grade badge */
.grade-A { color: #2ddb7c; font-size: 2.5rem; font-weight: 700; }
.grade-B { color: #4fb3ff; font-size: 2.5rem; font-weight: 700; }
.grade-C { color: #f5a623; font-size: 2.5rem; font-weight: 700; }
.grade-F { color: #ff4d6d; font-size: 2.5rem; font-weight: 700; }

/* Buttons */
button.primary {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    background: linear-gradient(135deg, #4ECDC4, #45B7D1) !important;
    color: #0a0e1a !important; border: none !important;
    border-radius: 8px !important; padding: 0.7rem 1.5rem !important;
    font-size: 0.9rem !important; transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.85 !important; }
button.secondary {
    font-family: 'Inter', sans-serif !important;
    background: #111827 !important; color: #4ECDC4 !important;
    border: 1px solid #4ECDC4 !important; border-radius: 8px !important;
}
button.stop {
    background: #1a0a0e !important; color: #ff4d6d !important;
    border: 1px solid #ff4d6d !important; border-radius: 8px !important;
}

/* Labels */
label span, .gradio-container label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #94a3b8 !important;
}

/* Inputs */
input[type=range] { -webkit-appearance: none; height: 4px; background: #1e293b; border-radius: 2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: #4ECDC4; cursor: pointer; border: 2px solid #0a0e1a;
}

/* Textarea */
textarea, .gradio-container textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; background: #0d1117 !important;
    color: #4ECDC4 !important; border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}

/* Markdown */
.gradio-container h2 { color: #4ECDC4 !important; font-size: 1.2rem !important; }
.gradio-container h3 { color: #45B7D1 !important; font-size: 1rem !important; }
.gradio-container p { color: #94a3b8 !important; }
table { width: 100%; border-collapse: collapse; }
th { background: #111827; color: #4ECDC4; font-size: 0.8rem;
     text-align: left; padding: 8px 12px; border-bottom: 1px solid #1e293b; }
td { padding: 8px 12px; border-bottom: 1px solid #0f172a; color: #e2e8f0; font-size: 0.88rem; }
blockquote { border-left: 3px solid #4ECDC4; padding-left: 1rem;
             color: #64748b !important; margin: 0.5rem 0; }

/* Accordion */
.gradio-container details { background: #111827; border: 1px solid #1e293b;
                             border-radius: 8px; padding: 0.75rem 1rem; }

footer { display: none !important; }
.gradio-container .block { background: transparent !important; border: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="The Swarm Architect") as demo:
    # ── Hero ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero">
        <div class="hero-title">The Swarm Architect</div>
        <div class="hero-sub">
            Watch 5 AI agents learn to coordinate — no coding required
        </div>
    </div>
    """)

    with gr.Tabs():
        # ═══════════════════════════════════════════════════════════════════
        # Tab 1 — The Mission
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("🏠 The Mission"):
            gr.HTML("""
            <div style="text-align:center; padding: 1rem 0 1.5rem;">
                <p style="color:#94a3b8; font-size:1.05rem; max-width:640px; margin:0 auto;">
                    Five AI agents are dropped into an empty space with five landmarks.
                    Their goal: each agent must cover a different landmark — without colliding.
                    They start knowing nothing. Through trial and error, they learn to cooperate.
                </p>
            </div>
            """)

            # 3-card explainer
            gr.HTML("""
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:2rem;">
                <div class="info-card">
                    <div class="info-card-icon">🤖</div>
                    <div class="info-card-title">Who are the agents?</div>
                    <div class="info-card-body">
                        Five independent AI programs. Each one can only see its own position,
                        velocity, and nearby agents. They cannot talk to each other —
                        they must learn to coordinate purely through experience.
                    </div>
                </div>
                <div class="info-card">
                    <div class="info-card-icon">🎯</div>
                    <div class="info-card-title">What is the task?</div>
                    <div class="info-card-body">
                        Cover all 5 landmarks simultaneously. The team loses points
                        for every uncovered landmark and every collision.
                        A perfect run means each agent claims one landmark and holds it.
                    </div>
                </div>
                <div class="info-card">
                    <div class="info-card-icon">🧠</div>
                    <div class="info-card-title">How do they learn?</div>
                    <div class="info-card-body">
                        Each agent runs thousands of trial episodes, getting a score at the end.
                        Good moves are reinforced, bad moves are forgotten.
                        This is called <strong>Reinforcement Learning</strong>.
                    </div>
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="background:#111827; border:1px solid #1e293b; border-radius:12px;
                        padding:1.5rem; margin-bottom:1rem;">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.5rem;">
                    🗺️ How this app works
                </div>
                <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:0.8rem;">
                    <div style="text-align:center;">
                        <div style="font-size:1.5rem;">👁️</div>
                        <div style="font-weight:600; color:#4ECDC4; font-size:0.9rem;">Step 1</div>
                        <div style="color:#94a3b8; font-size:0.83rem;">Watch trained agents coordinate in the <b>Watch the Swarm</b> tab</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.5rem;">⚡</div>
                        <div style="font-weight:600; color:#4ECDC4; font-size:0.9rem;">Step 2</div>
                        <div style="color:#94a3b8; font-size:0.83rem;">Train your own agents in the <b>Train Your Swarm</b> tab</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.5rem;">📊</div>
                        <div style="font-weight:600; color:#4ECDC4; font-size:0.9rem;">Step 3</div>
                        <div style="color:#94a3b8; font-size:0.83rem;">See how well they learned in the <b>Mission Report</b> tab</div>
                    </div>
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="background:linear-gradient(135deg,#0d1f1e,#0a1628);
                        border:1px solid #2d6a4f; border-radius:12px;
                        padding:1.2rem 1.5rem; display:flex; align-items:center; gap:1rem;">
                <div style="font-size:1.8rem;">💡</div>
                <div>
                    <div style="font-weight:600; color:#4ECDC4;">Did you know?</div>
                    <div style="color:#94a3b8; font-size:0.88rem;">
                        This technique — where agents learn by reward rather than explicit instructions —
                        is the same approach used to train AlphaGo, ChatGPT's alignment, and
                        autonomous drone swarms. You're looking at the same core idea.
                    </div>
                </div>
            </div>
            """)

        # ═══════════════════════════════════════════════════════════════════
        # Tab 2 — Watch the Swarm
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("👁️ Watch the Swarm"):
            gr.HTML("""
            <div style="padding:0.5rem 0 1.2rem;">
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">
                    See the agents in action
                </div>
                <div style="color:#64748b; font-size:0.88rem; margin-top:0.3rem;">
                    This runs the pre-trained agents through one episode. Watch how they
                    spread out and cover landmarks — or collide and fail!
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    ep_steps = gr.Slider(
                        10,
                        50,
                        value=30,
                        step=5,
                        label="How many steps to simulate",
                        info="More steps = longer episode (default 30 is a good starting point)",
                    )
                    btn_watch = gr.Button("▶ Watch the Agents", variant="primary")

                    gr.HTML("<div style='height:0.8rem'></div>")

                    gr.HTML("""
                    <div style="background:#111827; border:1px solid #1e293b;
                                border-radius:10px; padding:1rem;">
                        <div style="font-weight:600; color:#e2e8f0; font-size:0.88rem; margin-bottom:0.6rem;">
                            🔍 What to look for
                        </div>
                        <ul style="color:#94a3b8; font-size:0.82rem; line-height:1.8; margin:0; padding-left:1.2rem;">
                            <li>Each coloured dot is one agent</li>
                            <li>The X marks are the landmarks to cover</li>
                            <li>Good: agents spread to different landmarks</li>
                            <li>Bad: agents cluster together or crash</li>
                            <li>The score gets better as they learn not to collide</li>
                        </ul>
                    </div>
                    """)

                    ep_info = gr.Markdown("*Click Watch to run an episode.*")

                with gr.Column(scale=2):
                    ep_gif = gr.Image(
                        label="Live Episode Replay",
                        type="filepath",
                        height=380,
                    )

            btn_watch.click(
                cb_watch_demo,
                inputs=[ep_steps],
                outputs=[ep_gif, ep_info, gr.State()],
            )

        # ═══════════════════════════════════════════════════════════════════
        # Tab 3 — Train Your Swarm
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("⚡ Train Your Swarm"):
            gr.HTML("""
            <div style="padding:0.5rem 0 1.2rem;">
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">
                    Teach the agents from scratch
                </div>
                <div style="color:#64748b; font-size:0.88rem; margin-top:0.3rem;">
                    Training runs in the background. Check the
                    <strong style="color:#4ECDC4;">Mission Report</strong> tab and
                    click Refresh to see progress. Your trained agents will automatically
                    be used in the Watch tab.
                </div>
            </div>
            """)

            # Quick start
            gr.HTML("""
            <div style="background:linear-gradient(135deg,#0d1a2e,#0a1f1e);
                        border:1px solid #1e4a6e; border-radius:12px; padding:1.5rem;
                        margin-bottom:1rem;">
                <div style="font-weight:600; color:#4ECDC4; font-size:1rem; margin-bottom:0.4rem;">
                    ⚡ Quick Start
                </div>
                <div style="color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;">
                    Uses the best recommended settings. Good for a first run (takes ~2–3 mins).
                </div>
            </div>
            """)

            with gr.Row():
                btn_quick = gr.Button(
                    "⚡ Start Quick Training (200 episodes)", variant="primary"
                )
                btn_stop = gr.Button("⏹ Stop Training", variant="stop")

            quick_status = gr.Textbox(
                label="Training status",
                interactive=False,
                lines=2,
                placeholder="Click Start to begin…",
            )

            gr.HTML("<div style='height:1rem'></div>")

            # Advanced — collapsed accordion
            with gr.Accordion("🔧 Advanced Settings — customise training", open=False):
                gr.HTML("""
                <div style="color:#64748b; font-size:0.82rem; padding:0.5rem 0 1rem;">
                    Only change these if you know what you're doing.
                    Defaults are already tuned for this task.
                </div>
                """)
                with gr.Row():
                    with gr.Column():
                        adv_episodes = gr.Slider(
                            50,
                            2000,
                            value=300,
                            step=50,
                            label="Number of training episodes",
                            info="More episodes = better agents, but takes longer",
                        )
                        adv_lr = gr.Slider(
                            1e-5,
                            1e-2,
                            value=3e-4,
                            step=1e-5,
                            label="Learning speed",
                            info="How fast agents update their strategy (default 0.0003)",
                        )
                    with gr.Column():
                        adv_gamma = gr.Slider(
                            0.80,
                            0.999,
                            value=0.95,
                            step=0.005,
                            label="How far ahead agents think",
                            info="Higher = agents plan further into the future (default 0.95)",
                        )
                        adv_clip = gr.Slider(
                            0.05,
                            0.5,
                            value=0.2,
                            step=0.05,
                            label="How cautiously agents update",
                            info="Lower = more careful updates, less likely to forget (default 0.2)",
                        )
                        adv_rollout = gr.Slider(
                            64,
                            512,
                            value=128,
                            step=64,
                            label="Experience batch size",
                            info="Steps collected before each learning update (default 128)",
                        )

                btn_custom = gr.Button("▶ Start Custom Training", variant="secondary")
                custom_status = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=1,
                    placeholder="Status will appear here…",
                )

            btn_quick.click(cb_quick_train, outputs=[quick_status, gr.State()])
            btn_stop.click(cb_stop_training, outputs=[quick_status])
            btn_custom.click(
                cb_custom_train,
                inputs=[adv_episodes, adv_lr, adv_gamma, adv_clip, adv_rollout],
                outputs=[custom_status, gr.State()],
            )

            gr.HTML("""
            <div style="margin-top:1.5rem; background:#111827; border:1px solid #1e293b;
                        border-radius:12px; padding:1.2rem;">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.8rem;">❓ What happens during training?</div>
                <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; text-align:center;">
                    <div>
                        <div style="font-size:1.4rem;">🎲</div>
                        <div style="color:#4ECDC4; font-size:0.8rem; font-weight:600;">Episodes 1–50</div>
                        <div style="color:#64748b; font-size:0.78rem;">Random wandering — agents explore without a plan</div>
                    </div>
                    <div>
                        <div style="font-size:1.4rem;">📡</div>
                        <div style="color:#4ECDC4; font-size:0.8rem; font-weight:600;">Episodes 50–150</div>
                        <div style="color:#64748b; font-size:0.78rem;">Agents start spreading — collisions decrease</div>
                    </div>
                    <div>
                        <div style="font-size:1.4rem;">🤝</div>
                        <div style="color:#4ECDC4; font-size:0.8rem; font-weight:600;">Episodes 150–300</div>
                        <div style="color:#64748b; font-size:0.78rem;">Coordination emerges — agents learn their "role"</div>
                    </div>
                    <div>
                        <div style="font-size:1.4rem;">🏆</div>
                        <div style="color:#4ECDC4; font-size:0.8rem; font-weight:600;">Episodes 300+</div>
                        <div style="color:#64748b; font-size:0.78rem;">Stable policy — each agent reliably covers one landmark</div>
                    </div>
                </div>
            </div>
            """)

        # ═══════════════════════════════════════════════════════════════════
        # Tab 4 — Mission Report
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📈 Mission Report"):
            gr.HTML("""
            <div style="padding:0.5rem 0 1rem;">
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">
                    How well are your agents doing?
                </div>
                <div style="color:#64748b; font-size:0.88rem; margin-top:0.3rem;">
                    Click Refresh to pull the latest numbers from the training run.
                </div>
            </div>
            """)

            with gr.Row():
                btn_refresh = gr.Button("🔄 Refresh Report", variant="primary")
                with gr.Column(scale=3):
                    report_status = gr.Textbox(
                        label="Live status",
                        interactive=False,
                        lines=2,
                        placeholder="Status appears here after training starts…",
                    )

            report_summary = gr.Markdown(
                "*Start training, then refresh to see your results.*"
            )

            with gr.Row():
                training_plot = gr.Plot(label="Training Progress")
                reward_plot = gr.Plot(label="Reward History")

            btn_refresh.click(
                cb_refresh_report,
                outputs=[report_status, training_plot, reward_plot, report_summary],
            )

            gr.HTML("<div style='height:1rem'></div>")

            # Evaluation section
            gr.HTML("""
            <div style="background:#111827; border:1px solid #1e293b; border-radius:12px;
                        padding:1.2rem; margin-top:0.5rem;">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.4rem;">
                    🧪 Run a formal test
                </div>
                <div style="color:#64748b; font-size:0.85rem; margin-bottom:0.8rem;">
                    Test your trained agents over multiple episodes for a reliable grade.
                </div>
            </div>
            """)

            with gr.Row():
                n_eval = gr.Slider(
                    1,
                    10,
                    value=3,
                    step=1,
                    label="Number of test episodes",
                    info="More episodes = more reliable score",
                )
                btn_eval = gr.Button("📊 Test My Agents", variant="primary")

            eval_results = gr.Markdown("*Run a test to see your agents' grade.*")
            btn_eval.click(cb_evaluate, inputs=[n_eval], outputs=[eval_results])

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; color:#1e293b; font-size:0.75rem;
                padding:2rem 0 1rem; border-top:1px solid #111827; margin-top:1rem;">
        Built with Independent PPO · Multi-Agent Particle Environment · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
