"""
🤖 Maze Runner — RL Playground
An interactive, fun maze-solving playground powered by Reinforcement Learning.
Anyone can build a maze, pick a brain, and watch the bot learn to escape.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from agents.montecarlo import train_montecarlo
from agents.qlearning import train_qlearning
from agents.sarsa import train_sarsa
from maze.env import MazeEnv
from maze.generator import generate_dfs_maze, generate_open_maze
from viz.renderer import (
    make_qvalue_heatmap,
    make_race_chart,
    make_solution_gif,
    make_training_chart,
    score_run,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

ALGO_MAP = {
    "🧠 Q-Learning  (recommended)": "qlearning",
    "🎯 SARSA  (cautious)": "sarsa",
    "🎲 Monte Carlo  (explorer)": "montecarlo",
}

DIFFICULTY = {
    "🐣 Tiny  (5×5)": 5,
    "🐇 Small  (7×7)": 7,
    "🐢 Medium  (9×9)": 9,
    "🦊 Large  (13×13)": 13,
    "🐉 XL  (17×17)": 17,
}

MAZE_STYLE = {
    "🏰 Corridors  (DFS)": "dfs",
    "🌿 Open Field  (random walls)": "open",
}


def _make_env(size: int, style: str, seed: int) -> MazeEnv:
    if style == "dfs":
        grid = generate_dfs_maze(size, seed=seed)
    else:
        grid = generate_open_maze(size, wall_frac=0.18, seed=seed)
    return MazeEnv(grid)


def _train(
    env: MazeEnv,
    algo: str,
    episodes: int,
    alpha: float,
    gamma: float,
    decay: float,
    seed: int,
):
    fn = {
        "qlearning": train_qlearning,
        "sarsa": train_sarsa,
        "montecarlo": train_montecarlo,
    }[algo]
    return fn(env, episodes, alpha, gamma, decay, seed)


def _collect_path(env: MazeEnv, agent) -> list[tuple[int, ...]]:
    state, _ = env.reset()
    path: list[tuple[int, ...]] = [env.start]
    for _ in range(env.n_states * 3):
        action = agent.greedy_action(state)
        state, _, done, _, _ = env.step(action)
        path.append(env._from_state(state))
        if done:
            break
    return path


# ── Main Playground callback ──────────────────────────────────────────────────


def cb_solve(
    difficulty: str,
    maze_style: str,
    algo_label: str,
    episodes: int,
    alpha: float,
    gamma: float,
    decay: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    progress(0.05, desc="Building maze…")
    size = DIFFICULTY[difficulty]
    style = MAZE_STYLE[maze_style]
    algo = ALGO_MAP[algo_label]

    env = _make_env(size, style, int(seed))

    progress(0.15, desc=f"Training {algo_label.split('(')[0].strip()}…")
    agent, rewards = _train(
        env, algo, int(episodes), float(alpha), float(gamma), float(decay), int(seed)
    )

    progress(0.75, desc="Rendering solution…")
    env2 = _make_env(size, style, int(seed))
    gif_path = make_solution_gif(
        env2, agent, fps=7, label=algo_label.split("(")[0].strip()
    )

    progress(0.85, desc="Building charts…")
    env3 = _make_env(size, style, int(seed))
    path = _collect_path(env3, agent)
    sc = score_run(path, env3.goal, rewards, env3.n_states)

    train_fig = make_training_chart({algo_label.split("(")[0].strip(): rewards})

    env4 = _make_env(size, style, int(seed))
    heatmap_fig = make_qvalue_heatmap(env4, agent)

    stats_md = f"""
### {sc["grade"]} — {sc["verdict"]}

| | |
|---|---|
| **Solved** | {"✅ Yes" if sc["solved"] else "❌ No"} |
| **Steps taken** | `{sc["steps"]}` |
| **Efficiency score** | `{sc["efficiency"]}%` |
| **Avg reward (final 20%)** | `{sc["avg_reward"]:.1f}` |
| **Episodes trained** | `{int(episodes)}` |
| **Maze size** | `{env.shape[0]} × {env.shape[1]}` cells |

> **Efficiency** compares your bot's path length to the ideal shortest path.
> 100% = perfect. 0% = didn't make it.
"""
    progress(1.0, desc="Done!")
    return gif_path, train_fig, heatmap_fig, stats_md


# ── Algorithm Race callback ───────────────────────────────────────────────────


def cb_race(
    difficulty: str,
    maze_style: str,
    episodes: int,
    run_mc: bool,
    progress: gr.Progress = gr.Progress(),
):
    size = DIFFICULTY[difficulty]
    style = MAZE_STYLE[maze_style]
    seed = 77

    progress(0.1, desc="Training Q-Learning…")
    env_q = _make_env(size, style, seed)
    _, rq = _train(env_q, "qlearning", int(episodes), 0.1, 0.95, 0.995, seed)

    progress(0.4, desc="Training SARSA…")
    env_s = _make_env(size, style, seed)
    _, rs = _train(env_s, "sarsa", int(episodes), 0.1, 0.95, 0.995, seed)

    rc, name_c = None, ""
    if run_mc:
        progress(0.65, desc="Training Monte Carlo…")
        env_m = _make_env(size, style, seed)
        _, rc = _train(env_m, "montecarlo", int(episodes), 0.1, 0.95, 0.995, seed)
        name_c = "Monte Carlo"

    progress(0.9, desc="Building race chart…")
    fig = make_race_chart(rq, "Q-Learning", rs, "SARSA", rc, name_c)

    # Winner
    final_q = float(np.mean(rq[-max(1, len(rq) // 5) :]))
    final_s = float(np.mean(rs[-max(1, len(rs) // 5) :]))
    scores = {"Q-Learning": final_q, "SARSA": final_s}
    if rc:
        scores["Monte Carlo"] = float(np.mean(rc[-max(1, len(rc) // 5) :]))
    winner = max(scores, key=lambda k: scores[k])

    result_md = f"""
### 🏆 Race Result

| Algorithm | Final Score |
|---|---|
{"".join(f"| {'🥇 ' if k == winner else ''}{k} | `{v:.1f}` |" + chr(10) for k, v in scores.items())}

**Winner: {winner}** with a final average reward of `{scores[winner]:.1f}`

> All algorithms trained on the same maze with identical hyperparameters.
> Final score = average reward over the last 20% of episodes.
"""
    progress(1.0)
    return fig, result_md


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* Hero */
.hero { text-align:center; padding:2rem 1rem 1rem; }
.hero-title {
    font-size: clamp(2rem, 5vw, 3rem); font-weight: 700;
    background: linear-gradient(135deg, #3fb950, #58a6ff, #ffa657);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero-sub { color: #484f58; font-size: 0.95rem; letter-spacing:0.02em; }

/* Tabs */
.tab-nav { border-bottom: 1px solid #21262d !important; background: transparent !important; }
.tab-nav button {
    font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 500 !important; color: #484f58 !important;
    background: transparent !important; border: none !important;
    padding: 0.7rem 1.1rem !important;
}
.tab-nav button.selected { color: #3fb950 !important; border-bottom: 2px solid #3fb950 !important; }
.tab-nav button:hover { color: #8b949e !important; }

/* Cards */
.info-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1.1rem;
}
.info-card-icon { font-size: 1.8rem; margin-bottom:0.4rem; }
.info-card-title { font-weight: 600; color: #e6edf3; font-size: 0.95rem; margin-bottom:0.3rem; }
.info-card-body { color: #8b949e; font-size: 0.83rem; line-height: 1.6; }

/* Algo cards */
.algo-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem; margin-bottom: 0.5rem;
}
.algo-name { font-weight: 600; color: #e6edf3; font-size: 0.9rem; }
.algo-desc { color: #8b949e; font-size: 0.8rem; line-height:1.5; margin-top:0.2rem; }
.algo-tag {
    display: inline-block; font-size: 0.68rem; padding: 2px 8px;
    border-radius: 20px; margin-top: 0.4rem;
}
.tag-green  { background:#0d2d17; color:#3fb950; border:1px solid #1a5c2e; }
.tag-blue   { background:#0d1f38; color:#58a6ff; border:1px solid #1a4a7a; }
.tag-orange { background:#2d1c06; color:#ffa657; border:1px solid #5c3a12; }

/* Grade badge */
.grade-badge {
    display:inline-block; font-size:2.5rem; font-weight:700;
    font-family:'JetBrains Mono',monospace;
}

/* Buttons */
button.primary {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 6px !important; font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.85 !important; }
button.secondary {
    background: #161b22 !important; color: #58a6ff !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
}
button.stop {
    background: #1c0d0d !important; color: #f78166 !important;
    border: 1px solid #6e2b2b !important; border-radius: 6px !important;
}

/* Labels */
label span { font-family:'Inter',sans-serif !important;
             font-size:0.82rem !important; color:#8b949e !important; }

/* Slider */
input[type=range] { -webkit-appearance:none; height:4px;
                    background:#21262d; border-radius:2px; }
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:16px; height:16px;
    border-radius:50%; background:#3fb950; cursor:pointer; border:2px solid #0d1117;
}

/* Textarea */
textarea { font-family:'JetBrains Mono',monospace !important;
           font-size:0.82rem !important; background:#0d1117 !important;
           color:#3fb950 !important; border:1px solid #21262d !important;
           border-radius:6px !important; }

/* Markdown */
.gradio-container h2 { color: #3fb950 !important; }
.gradio-container h3 { color: #58a6ff !important; }
.gradio-container p  { color: #8b949e !important; }
table { width:100%; border-collapse:collapse; }
th { background:#161b22; color:#3fb950; font-size:0.78rem;
     text-align:left; padding:8px 12px; border-bottom:1px solid #21262d; }
td { padding:8px 12px; border-bottom:1px solid #0d1117;
     color:#e6edf3; font-size:0.85rem; }
blockquote { border-left:3px solid #3fb950; padding-left:1rem;
             color:#484f58 !important; margin:0.5rem 0; }

footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="🤖 Maze Runner — RL Playground") as demo:
    gr.HTML("""
    <div class="hero">
        <div class="hero-title">🤖 Maze Runner</div>
        <div class="hero-sub">An AI that learns to escape mazes — watch it happen in real time</div>
    </div>
    """)

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Welcome
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🏠 Welcome"):
            gr.HTML("""
            <div style="text-align:center;padding:0.5rem 0 1.5rem;">
                <p style="color:#8b949e;font-size:1rem;max-width:580px;margin:0 auto;">
                    A tiny AI robot is dropped into a maze. It knows nothing.
                    Through thousands of attempts — hitting walls, finding dead ends,
                    occasionally stumbling upon the exit — it slowly builds a mental map
                    and learns the perfect escape route.
                </p>
            </div>
            """)

            gr.HTML("""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">
                <div class="info-card">
                    <div class="info-card-icon">🗺️</div>
                    <div class="info-card-title">The Maze</div>
                    <div class="info-card-body">
                        A grid of corridors and walls. The bot starts at
                        <strong style="color:#58a6ff">S</strong> and must reach
                        <strong style="color:#f78166">G</strong>.
                        It can only see its own position — no map, no cheating.
                    </div>
                </div>
                <div class="info-card">
                    <div class="info-card-icon">🤖</div>
                    <div class="info-card-title">The Bot</div>
                    <div class="info-card-body">
                        At each step it chooses: go up, down, left, or right.
                        Hit a wall? Penalty. Reach the goal? Big reward!
                        It remembers what worked and what didn't.
                    </div>
                </div>
                <div class="info-card">
                    <div class="info-card-icon">🧠</div>
                    <div class="info-card-title">The Learning</div>
                    <div class="info-card-body">
                        Each attempt updates a "score table" for every
                        position and move. After enough tries, the bot
                        always picks the move with the highest score — the optimal path.
                    </div>
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
                <div style="font-weight:600;color:#e6edf3;margin-bottom:1rem;">🧠 Choose your Bot's Brain</div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;">
                    <div class="algo-card">
                        <div class="algo-name">Q-Learning</div>
                        <div class="algo-desc">
                            Updates its score table <em>immediately</em> after every step.
                            Fast learner. Best for most mazes.
                        </div>
                        <span class="algo-tag tag-green">⚡ Recommended</span>
                    </div>
                    <div class="algo-card">
                        <div class="algo-name">SARSA</div>
                        <div class="algo-desc">
                            Updates based on the move it <em>actually took</em> next,
                            not just the best possible. More cautious, avoids risky paths.
                        </div>
                        <span class="algo-tag tag-blue">🎯 Cautious</span>
                    </div>
                    <div class="algo-card">
                        <div class="algo-name">Monte Carlo</div>
                        <div class="algo-desc">
                            Plays out the <em>entire episode</em> first, then
                            updates everything at once. Needs more episodes to converge.
                        </div>
                        <span class="algo-tag tag-orange">🎲 Explorer</span>
                    </div>
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div style="background:#0d2d17;border:1px solid #1a5c2e;border-radius:10px;padding:1rem;">
                    <div style="font-weight:600;color:#3fb950;margin-bottom:0.4rem;">🗺️ How to use this app</div>
                    <ol style="color:#8b949e;font-size:0.85rem;line-height:1.8;margin:0;padding-left:1.2rem;">
                        <li>Go to <strong style="color:#e6edf3">🎮 Playground</strong> tab</li>
                        <li>Pick a difficulty and maze style</li>
                        <li>Choose a brain and hit <strong style="color:#3fb950">Train & Watch!</strong></li>
                        <li>Watch the animated replay</li>
                        <li>Try <strong style="color:#e6edf3">🏁 Algorithm Race</strong> to compare all three</li>
                    </ol>
                </div>
                <div style="background:#0d1f38;border:1px solid #1a4a7a;border-radius:10px;padding:1rem;">
                    <div style="font-weight:600;color:#58a6ff;margin-bottom:0.4rem;">💡 Fun facts</div>
                    <ul style="color:#8b949e;font-size:0.85rem;line-height:1.8;margin:0;padding-left:1.2rem;">
                        <li>This same idea trains robots, game AIs, and self-driving cars</li>
                        <li>DeepMind's AlphaGo used a version of Q-Learning</li>
                        <li>A 17×17 maze has 289 possible positions to learn</li>
                        <li>The bot gets worse before it gets better — that's normal!</li>
                    </ul>
                </div>
            </div>
            """)

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Playground
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🎮 Playground"):
            gr.HTML("""
            <div style="padding:0.3rem 0 1rem;">
                <div style="font-size:1.05rem;font-weight:600;color:#e6edf3;">
                    Build a maze, pick a brain, watch it learn
                </div>
                <div style="color:#484f58;font-size:0.85rem;margin-top:0.2rem;">
                    The animated replay shows the final learned path after training.
                </div>
            </div>
            """)

            with gr.Row():
                # ── Controls ──────────────────────────────────────────────
                with gr.Column(scale=1, min_width=300):
                    gr.HTML(
                        '<div style="font-size:0.75rem;font-weight:600;color:#484f58;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">🗺️ MAZE SETUP</div>'
                    )

                    difficulty = gr.Radio(
                        list(DIFFICULTY.keys()),
                        value="🐢 Medium  (9×9)",
                        label="Difficulty",
                    )
                    maze_style = gr.Radio(
                        list(MAZE_STYLE.keys()),
                        value="🏰 Corridors  (DFS)",
                        label="Maze style",
                        info="Corridors = proper winding paths · Open = random walls",
                    )

                    gr.HTML(
                        '<div style="font-size:0.75rem;font-weight:600;color:#484f58;text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.5rem;">🧠 BOT BRAIN</div>'
                    )

                    algo = gr.Radio(
                        list(ALGO_MAP.keys()),
                        value="🧠 Q-Learning  (recommended)",
                        label="Algorithm",
                    )

                    gr.HTML(
                        '<div style="font-size:0.75rem;font-weight:600;color:#484f58;text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.5rem;">⚙️ TRAINING</div>'
                    )

                    episodes = gr.Slider(
                        100,
                        3000,
                        value=800,
                        step=100,
                        label="Training episodes",
                        info="More = smarter bot, but slower",
                    )

                    with gr.Accordion("🔬 Advanced settings", open=False):
                        alpha = gr.Slider(
                            0.01, 0.5, value=0.1, step=0.01, label="Learning speed (α)"
                        )
                        gamma = gr.Slider(
                            0.5,
                            0.99,
                            value=0.95,
                            step=0.01,
                            label="Future planning (γ)",
                        )
                        decay = gr.Slider(
                            0.90,
                            0.999,
                            value=0.995,
                            step=0.001,
                            label="Exploration decay",
                        )
                        seed = gr.Slider(0, 100, value=42, step=1, label="Random seed")

                    btn_solve = gr.Button("🚀 Train & Watch!", variant="primary")

                # ── Outputs ───────────────────────────────────────────────
                with gr.Column(scale=2):
                    play_stats = gr.Markdown(
                        "*Configure your maze and hit Train & Watch!*"
                    )

                    with gr.Row():
                        play_gif = gr.Image(
                            label="🎬 Bot solving the maze (animated)",
                            type="filepath",
                            height=360,
                        )

            with gr.Row():
                play_train_fig = gr.Plot(label="📈 Training progress")
                play_heatmap = gr.Plot(label="🌡️ Q-value map (what the bot learned)")

            # hidden state defaults for advanced
            alpha_h = gr.State(0.1)
            gamma_h = gr.State(0.95)
            decay_h = gr.State(0.995)
            seed_h = gr.State(42)

            btn_solve.click(
                cb_solve,
                inputs=[
                    difficulty,
                    maze_style,
                    algo,
                    episodes,
                    alpha,
                    gamma,
                    decay,
                    seed,
                ],
                outputs=[play_gif, play_train_fig, play_heatmap, play_stats],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Algorithm Race
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🏁 Algorithm Race"):
            gr.HTML("""
            <div style="padding:0.3rem 0 1rem;">
                <div style="font-size:1.05rem;font-weight:600;color:#e6edf3;">
                    Head-to-head: which brain learns fastest?
                </div>
                <div style="color:#484f58;font-size:0.85rem;margin-top:0.2rem;">
                    All algorithms train on the same maze with identical settings —
                    the only variable is the learning strategy.
                </div>
            </div>
            """)

            gr.HTML("""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-bottom:1rem;">
                <div style="background:#0d2d17;border:1px solid #1a5c2e;border-radius:8px;padding:0.8rem;text-align:center;">
                    <div style="color:#3fb950;font-size:1.2rem;font-weight:700;">Q-Learning</div>
                    <div style="color:#484f58;font-size:0.78rem;margin-top:0.2rem;">Off-policy · Fast update · Optimistic</div>
                </div>
                <div style="background:#0d1f38;border:1px solid #1a4a7a;border-radius:8px;padding:0.8rem;text-align:center;">
                    <div style="color:#58a6ff;font-size:1.2rem;font-weight:700;">SARSA</div>
                    <div style="color:#484f58;font-size:0.78rem;margin-top:0.2rem;">On-policy · Careful update · Conservative</div>
                </div>
                <div style="background:#2d1c06;border:1px solid #5c3a12;border-radius:8px;padding:0.8rem;text-align:center;">
                    <div style="color:#ffa657;font-size:1.2rem;font-weight:700;">Monte Carlo</div>
                    <div style="color:#484f58;font-size:0.78rem;margin-top:0.2rem;">Episodic · Full return · Unbiased</div>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    race_diff = gr.Radio(
                        list(DIFFICULTY.keys()),
                        value="🐢 Medium  (9×9)",
                        label="Maze difficulty",
                    )
                    race_style = gr.Radio(
                        list(MAZE_STYLE.keys()),
                        value="🏰 Corridors  (DFS)",
                        label="Maze style",
                    )
                    race_eps = gr.Slider(
                        200, 2000, value=600, step=100, label="Episodes per algorithm"
                    )
                    race_mc = gr.Checkbox(
                        label="Include Monte Carlo (slower)", value=True
                    )
                    btn_race = gr.Button("🏁 Start Race!", variant="primary")

                with gr.Column(scale=2):
                    race_result = gr.Markdown(
                        "*Click Start Race to run the comparison.*"
                    )

            race_fig = gr.Plot(label="Race Results")

            btn_race.click(
                cb_race,
                inputs=[race_diff, race_style, race_eps, race_mc],
                outputs=[race_fig, race_result],
            )

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — How it Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🧠 How it Works"):
            gr.HTML("""
            <div style="max-width:700px;margin:0 auto;padding:1rem 0;">

            <h2 style="color:#3fb950;font-size:1.3rem;margin-bottom:0.3rem;">The Big Idea</h2>
            <p style="color:#8b949e;line-height:1.7;">
                The bot doesn't know anything about the maze at the start. It just knows
                4 possible moves and gets a number (reward) after each step.
                <strong style="color:#e6edf3">Negative number = bad move. Positive = good move.</strong>
                The goal: find the sequence of moves that gets the most reward.
            </p>

            <h2 style="color:#3fb950;font-size:1.3rem;margin:1.2rem 0 0.3rem;">The Score Table (Q-Table)</h2>
            <p style="color:#8b949e;line-height:1.7;">
                The bot keeps a table with one row per maze cell and 4 columns (one per direction).
                Each entry stores <em>how good it thinks that move is from that cell</em>.
                At the start, everything is 0. After training, the table holds the bot's
                entire learned strategy. The Q-value heatmap in the Playground shows this table visually.
            </p>

            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:1rem;margin:1rem 0;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#3fb950;">
Q[current_cell][move] += learning_speed × (<br>
&nbsp;&nbsp;reward_got + future_discount × best_Q[next_cell] − Q[current_cell][move]<br>
)
            </div>

            <h2 style="color:#3fb950;font-size:1.3rem;margin:1.2rem 0 0.3rem;">Exploration vs Exploitation</h2>
            <p style="color:#8b949e;line-height:1.7;">
                Early in training, the bot tries <strong style="color:#e6edf3">random moves</strong> (exploration)
                — it doesn't know enough to trust its table yet. Over time, it relies more on what
                it's learned (exploitation). This is controlled by <strong style="color:#e6edf3">epsilon (ε)</strong>,
                which starts near 1.0 (100% random) and decays toward 0 (always use best known move).
            </p>

            <h2 style="color:#3fb950;font-size:1.3rem;margin:1.2rem 0 0.3rem;">Why does reward go negative first?</h2>
            <p style="color:#8b949e;line-height:1.7;">
                Each step costs −1 (time penalty) and hitting a wall costs −5.
                A random bot hits a <em>lot</em> of walls and takes forever to find the exit,
                so early rewards are very negative. As it learns, fewer walls are hit and
                the path shortens — reward climbs toward 0 and eventually turns positive when
                it reliably reaches the goal.
            </p>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:1.2rem;">
                <div style="background:#0d2d17;border:1px solid #1a5c2e;border-radius:8px;padding:0.9rem;">
                    <div style="color:#3fb950;font-weight:600;margin-bottom:0.4rem;">Q-Learning vs SARSA</div>
                    <div style="color:#8b949e;font-size:0.82rem;line-height:1.6;">
                        Q-Learning always updates toward the <em>best possible</em> next action —
                        even if it wouldn't actually take that action. SARSA updates toward
                        the action it <em>will actually take</em>. This makes SARSA more cautious near walls.
                    </div>
                </div>
                <div style="background:#2d1c06;border:1px solid #5c3a12;border-radius:8px;padding:0.9rem;">
                    <div style="color:#ffa657;font-weight:600;margin-bottom:0.4rem;">Why Monte Carlo is slow</div>
                    <div style="color:#8b949e;font-size:0.82rem;line-height:1.6;">
                        MC waits until the episode <em>ends</em> before updating any scores.
                        On large mazes where early episodes never reach the goal,
                        it gets zero signal for a long time. But once it starts solving,
                        its estimates are very accurate.
                    </div>
                </div>
            </div>

            </div>
            """)

    gr.HTML("""
    <div style="text-align:center;color:#21262d;font-size:0.75rem;
                padding:1.5rem 0 0.5rem;border-top:1px solid #161b22;margin-top:1rem;">
        Built with Q-Learning · SARSA · Monte Carlo · Gymnasium · Gradio
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
