"""
Digital Calligrapher — RLHF Aesthetic Learning Engine

A Zen-inspired calligraphy playground where an AI learns your personal
aesthetic preferences through pairwise comparisons (RLHF).

Algorithm: Bradley-Terry reward model with gradient descent.
The reward model learns a 6-dimensional weight vector representing
your aesthetic preferences for stroke complexity, smoothness,
pressure, curvature, length, and randomness.
"""

import io
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from PIL import Image

# ── Ink palette ───────────────────────────────────────────────────────────────

INK_STYLES = {
    "Sumi Ink":   {"stroke": "#1a1a1a", "bg": "#f4f1ea", "accent": "#2d3436"},
    "Vermillion": {"stroke": "#c0392b", "bg": "#fdf6f0", "accent": "#922b21"},
    "Indigo":     {"stroke": "#1a3a5c", "bg": "#f0f4f8", "accent": "#0d2137"},
    "Gold":       {"stroke": "#b7860b", "bg": "#fdf9f0", "accent": "#8a6509"},
    "Jade":       {"stroke": "#1a6b3c", "bg": "#f0fdf4", "accent": "#0f4a28"},
    "Charcoal":   {"stroke": "#4a4a4a", "bg": "#f8f7f5", "accent": "#2a2a2a"},
    "Midnight":   {"stroke": "#2c3e50", "bg": "#f5f6fa", "accent": "#1a252f"},
}

FEATURE_NAMES = ["Complexity", "Smoothness", "Pressure", "Curvature", "Length", "Randomness"]

# ── RLHF Reward Model ─────────────────────────────────────────────────────────

class RewardModel:
    """
    Linear reward model trained with Bradley-Terry pairwise comparisons.
    P(A preferred over B) = sigmoid(R(A) - R(B))
    Trained via cross-entropy gradient descent.
    """

    def __init__(self, feature_dim: int = 6, lr: float = 0.4):
        self.weights     = np.zeros(feature_dim)
        self.lr          = lr
        self.vote_count  = 0
        self.loss_history: list[float] = []
        self.weight_history: list[np.ndarray] = []

    def score(self, features: np.ndarray) -> float:
        return float(np.dot(features, self.weights))

    def update(self, preferred: np.ndarray, rejected: np.ndarray):
        diff        = self.score(preferred) - self.score(rejected)
        p_wrong     = 1.0 / (1.0 + np.exp(diff))          # sigmoid(-diff)
        loss        = -np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-10)
        gradient    = p_wrong * (preferred - rejected)
        self.weights += self.lr * gradient
        self.vote_count += 1
        self.loss_history.append(float(loss))
        self.weight_history.append(self.weights.copy())

    @property
    def confidence(self) -> float:
        """Pseudo-confidence: how much the model has converged (0-100%)."""
        if self.vote_count < 2:
            return 0.0
        recent = self.loss_history[-min(10, len(self.loss_history)):]
        spread = np.std(recent)
        return float(np.clip(100 * (1 - spread / (np.mean(np.abs(recent)) + 1e-8)), 0, 100))

    def optimal_params(self, n_restarts: int = 500) -> np.ndarray:
        """
        Random-restart hill climbing to find features that maximise learned reward.
        This is the real optimisation — not just weight clipping.
        """
        best_score   = -np.inf
        best_params  = np.random.rand(6)

        for _ in range(n_restarts):
            candidate = np.random.rand(6)
            s = self.score(candidate)
            if s > best_score:
                best_score  = s
                best_params = candidate.copy()

        # Fine-tune with gradient ascent
        params = best_params.copy()
        for _ in range(200):
            grad   = self.weights                    # gradient of linear reward
            params = np.clip(params + 0.01 * grad, 0, 1)

        return np.clip(params, 0, 1)


# ── Stroke generation ─────────────────────────────────────────────────────────

def generate_stroke(params: np.ndarray, seed: int | None = None) -> tuple:
    """
    Generate a parametric calligraphic stroke.

    params = [complexity, smoothness, pressure_peak, curvature, length, randomness]
    All values in [0, 1].

    Now all 6 parameters are actively used:
      - complexity  → number of control points (3-10)
      - smoothness  → spline degree (2-4)
      - pressure    → peak line width multiplier
      - curvature   → sine wave amplitude applied to y
      - length      → horizontal span of stroke (0.3-1.0)
      - randomness  → jitter added to control points
    """
    rng = np.random.default_rng(seed)

    complexity  = int(3 + params[0] * 7)     # 3-10 control points
    smoothness  = int(2 + params[1] * 2)      # spline degree 2-4
    peak        = 1.0 + params[2] * 4.5       # max linewidth 1-5.5
    curvature   = params[3] * 2.2             # y-offset amplitude
    length      = 0.3 + params[4] * 0.7       # horizontal span 0.3-1.0
    randomness  = params[5] * 0.15            # point jitter magnitude

    k = min(smoothness, complexity - 1)       # degree must be < n_points

    # Control points
    t = np.linspace(0, 1, complexity)
    x = np.linspace(0, length, complexity) + rng.uniform(-randomness, randomness, complexity)
    y = rng.uniform(0.3, 0.7, complexity)     # base midline
    y += np.sin(x / length * np.pi) * curvature
    y += rng.uniform(-randomness, randomness, complexity)

    # Sort by x for monotone interpolation
    order = np.argsort(x)
    x, y  = x[order], y[order]

    # Deduplicate x (required for spline)
    _, unique = np.unique(x, return_index=True)
    x, y, t_u = x[unique], y[unique], t[unique]
    k = min(k, len(x) - 1)

    try:
        t_dense = np.linspace(0, t_u[-1], 200)
        sx = make_interp_spline(t_u, x, k=k)(t_dense)
        sy = make_interp_spline(t_u, y, k=k)(t_dense)
    except Exception:
        sx = np.linspace(0, length, 200)
        sy = np.full(200, 0.5)

    # Tapered pressure curve (thin at ends, thick in middle)
    pressure = np.sin(np.linspace(0, np.pi, 200)) * peak

    return sx, sy, pressure


def render_stroke(
    x: np.ndarray, y: np.ndarray, p: np.ndarray,
    ink: dict,
    figsize: float = 5,
    dpi: int = 110,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    fig.patch.set_facecolor(ink["bg"])
    ax.set_facecolor(ink["bg"])

    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2],
                color=ink["stroke"], linewidth=p[i],
                solid_capstyle="round", alpha=0.88)

    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=ink["bg"],
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def render_comparison_pair(
    feat_a: np.ndarray, feat_b: np.ndarray, ink: dict
) -> tuple[Image.Image, Image.Image]:
    sa = generate_stroke(feat_a, seed=int(feat_a[0] * 999))
    sb = generate_stroke(feat_b, seed=int(feat_b[0] * 1337))
    return render_stroke(*sa, ink), render_stroke(*sb, ink)


# ── Analytics charts ──────────────────────────────────────────────────────────

PARCHMENT = "#f4f1ea"
INK_DARK  = "#2d3436"
MUTED     = "#9b9080"
WARM_GREEN = "#1D9E75"
WARM_RED   = "#c0392b"


def _ax_style(ax):
    ax.set_facecolor(PARCHMENT)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#ddd8cc")
    ax.grid(color="#e8e4da", linewidth=0.5, linestyle="--", alpha=0.8)


def make_learning_curve(model: RewardModel) -> Image.Image:
    """Smoothed cross-entropy loss over votes."""
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    fig.patch.set_facecolor(PARCHMENT)
    _ax_style(ax)

    if len(model.loss_history) < 2:
        ax.text(0.5, 0.5, "Cast more votes to see the learning curve",
                ha="center", va="center", color=MUTED,
                fontsize=11, style="italic", transform=ax.transAxes)
        ax.axis("off")
    else:
        lh = np.array(model.loss_history)
        ax.fill_between(range(len(lh)), lh, alpha=0.15, color=INK_DARK)
        ax.plot(lh, color=MUTED, linewidth=0.8, alpha=0.5)
        if len(lh) > 5:
            k = max(3, len(lh) // 8)
            smooth = np.convolve(lh, np.ones(k) / k, "valid")
            ax.plot(range(k - 1, len(lh)), smooth, color=INK_DARK, linewidth=2.0)
        ax.set_xlabel("Vote", color=MUTED, fontsize=8)
        ax.set_ylabel("Loss", color=MUTED, fontsize=8)
        ax.set_title("Reward Model Learning Curve", color=INK_DARK,
                     fontsize=10, fontfamily="serif", style="italic", pad=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PARCHMENT, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_radar_chart(model: RewardModel) -> Image.Image:
    """Polar 'aesthetic DNA' chart showing learned weight profile."""
    n = len(FEATURE_NAMES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=110,
                           subplot_kw={"polar": True})
    fig.patch.set_facecolor(PARCHMENT)
    ax.set_facecolor(PARCHMENT)

    if model.vote_count > 0:
        # Normalise weights to [0,1] for display
        w = model.weights.copy()
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        values = w_norm.tolist() + [w_norm[0]]

        ax.fill(angles, values, color=INK_DARK, alpha=0.18)
        ax.plot(angles, values, color=INK_DARK, linewidth=2.0)
        for angle, val, name in zip(angles[:-1], w_norm, FEATURE_NAMES):
            col = WARM_GREEN if model.weights[FEATURE_NAMES.index(name)] >= 0 else WARM_RED
            ax.scatter(angle, val, color=col, s=40, zorder=5)
    else:
        values = [0.5] * (n + 1)
        ax.fill(angles, values, color=MUTED, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURE_NAMES, color=INK_DARK, fontsize=8.5)
    ax.set_yticklabels([])
    ax.grid(color="#ddd8cc", linewidth=0.6)
    ax.set_title("Your Aesthetic Profile", color=INK_DARK, pad=14,
                 fontsize=11, fontfamily="serif", style="italic")
    ax.spines["polar"].set_color("#ddd8cc")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PARCHMENT, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_weight_bar(model: RewardModel) -> Image.Image:
    """Horizontal bar chart of raw learned weights."""
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=110)
    fig.patch.set_facecolor(PARCHMENT)
    _ax_style(ax)
    ax.set_facecolor(PARCHMENT)

    labels = FEATURE_NAMES
    w = model.weights if model.vote_count > 0 else np.zeros(6)
    colors = [WARM_GREEN if v >= 0 else WARM_RED for v in w]

    bars = ax.barh(labels, w, color=colors, edgecolor=PARCHMENT, linewidth=0.5, alpha=0.8)
    ax.axvline(0, color="#ccc8ba", linewidth=1.2)
    for bar, val in zip(bars, w):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                color=INK_DARK, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Weight", color=MUTED, fontsize=8)
    ax.set_title("Learned Preference Weights", color=INK_DARK, pad=8,
                 fontsize=10, fontfamily="serif", style="italic")
    ax.tick_params(axis="y", colors=INK_DARK, labelsize=8.5)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PARCHMENT, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── App state ─────────────────────────────────────────────────────────────────
# All state is passed through gr.State — no globals except the ink palette.


def _initial_state(ink_name: str = "Sumi Ink") -> dict:
    model = RewardModel()
    feat_a, feat_b = np.random.rand(6), np.random.rand(6)
    return {
        "model":   model,
        "feat_a":  feat_a,
        "feat_b":  feat_b,
        "ink":     ink_name,
        "gallery": [],      # list of PIL Images (max 6)
    }


def _status_html(state: dict) -> str:
    model = state["model"]
    if model.vote_count == 0:
        return """
<div style='text-align:center;padding:1.5rem;color:#9b9080;font-style:italic;font-family:serif;font-size:1rem;'>
Cast your first vote to begin shaping your aesthetic profile.
</div>"""

    w = model.weights
    conf = model.confidence
    bars = "".join(
        f"""<div style='display:flex;justify-content:space-between;padding:4px 0;
                border-bottom:1px dashed #e8e4da;font-size:0.82rem;'>
              <span style='color:#5a5550;'>{n}</span>
              <span style='color:{"#1D9E75" if v>=0 else "#c0392b"};
                    font-family:monospace;font-weight:600;'>{v:+.3f}</span>
            </div>"""
        for n, v in zip(FEATURE_NAMES, w)
    )
    return f"""
<div style='font-family:serif;'>
  <div style='font-size:0.78rem;color:#9b9080;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;'>
    Votes: <strong style='color:#2d3436'>{model.vote_count}</strong>
    &nbsp;·&nbsp; Confidence: <strong style='color:#1D9E75'>{conf:.0f}%</strong>
  </div>
  {bars}
</div>"""


# ── Callbacks ─────────────────────────────────────────────────────────────────

def cb_init(state: dict):
    ink = INK_STYLES[state["ink"]]
    img_a, img_b = render_comparison_pair(state["feat_a"], state["feat_b"], ink)
    return img_a, img_b, _status_html(state), state


def cb_vote(choice: str, state: dict):
    model = state["model"]
    if choice == "A":
        model.update(state["feat_a"], state["feat_b"])
    else:
        model.update(state["feat_b"], state["feat_a"])

    # Generate new pair — after enough votes, bias one toward the learned optimum
    feat_a = np.random.rand(6)
    if model.vote_count >= 3:
        # Exploit: one stroke near the current optimum
        noise  = np.random.normal(0, 0.12, 6)
        feat_b = np.clip(model.weights + noise, 0, 1)
    else:
        feat_b = np.random.rand(6)

    state["feat_a"] = feat_a
    state["feat_b"] = feat_b
    ink = INK_STYLES[state["ink"]]
    img_a, img_b = render_comparison_pair(feat_a, feat_b, ink)
    return img_a, img_b, _status_html(state), state


def cb_skip(state: dict):
    state["feat_a"] = np.random.rand(6)
    state["feat_b"] = np.random.rand(6)
    ink = INK_STYLES[state["ink"]]
    img_a, img_b = render_comparison_pair(state["feat_a"], state["feat_b"], ink)
    return img_a, img_b, state


def cb_change_ink(ink_name: str, state: dict):
    state["ink"] = ink_name
    ink = INK_STYLES[ink_name]
    img_a, img_b = render_comparison_pair(state["feat_a"], state["feat_b"], ink)
    return img_a, img_b, state


def cb_generate_masterpiece(state: dict):
    """Real optimisation: hill-climbing maximises learned reward, then renders."""
    model = state["model"]
    if model.vote_count == 0:
        # No preferences learned yet — use random
        best_feat = np.random.rand(6)
        msg = "Cast some votes first for a personalised masterpiece!"
    else:
        best_feat = model.optimal_params(n_restarts=600)
        msg = (f"Optimised for your aesthetic "
               f"(reward score: {model.score(best_feat):+.3f})")

    ink = INK_STYLES[state["ink"]]
    sx, sy, sp = generate_stroke(best_feat, seed=42)
    img = render_stroke(sx, sy, sp, ink, figsize=6, dpi=130)

    # Save to gallery (max 6)
    gallery = state.get("gallery", [])
    gallery = [img] + gallery
    gallery = gallery[:6]
    state["gallery"] = gallery

    return img, msg, state


def cb_refresh_charts(state: dict):
    model = state["model"]
    lc  = make_learning_curve(model)
    rad = make_radar_chart(model)
    wb  = make_weight_bar(model)
    return lc, rad, wb


def cb_reset(state: dict):
    state["model"] = RewardModel()
    state["feat_a"] = np.random.rand(6)
    state["feat_b"] = np.random.rand(6)
    state["gallery"] = []
    ink = INK_STYLES[state["ink"]]
    img_a, img_b = render_comparison_pair(state["feat_a"], state["feat_b"], ink)
    empty_lc  = make_learning_curve(state["model"])
    empty_rad = make_radar_chart(state["model"])
    empty_wb  = make_weight_bar(state["model"])
    return (img_a, img_b, _status_html(state),
            empty_lc, empty_rad, empty_wb, state)


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background-color: #f4f1ea !important;
    color: #2d3436 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }

/* Header */
.callig-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #e0dcd0;
    margin-bottom: 1.5rem;
}
.callig-title {
    font-family: 'EB Garamond', serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 400; letter-spacing: 0.12em;
    color: #1a1a1a; margin: 0 0 0.3rem;
    text-transform: uppercase;
}
.callig-sub {
    font-family: 'EB Garamond', serif;
    font-style: italic; font-size: 1.05rem; color: #9b9080;
}
.callig-badges { display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap;margin-top:0.8rem; }
.badge {
    font-family:'Inter',sans-serif; font-size:0.65rem; letter-spacing:0.1em;
    padding:3px 10px; border-radius:20px; text-transform:uppercase;
    background:rgba(45,52,54,0.06); color:#5a5550; border:1px solid #ddd8cc;
}

/* Tabs */
.tab-nav { border-bottom:1px solid #e0dcd0 !important; background:transparent !important; }
.tab-nav button {
    font-family:'EB Garamond',serif !important; font-size:1rem !important;
    font-style:italic !important; color:#9b9080 !important;
    background:transparent !important; border:none !important;
    padding:0.6rem 1.2rem !important;
}
.tab-nav button.selected {
    color:#1a1a1a !important;
    border-bottom:2px solid #1a1a1a !important;
    font-style:normal !important;
}

/* Stroke cards */
.stroke-card {
    background: #fff;
    border: 1px solid #e8e4da;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: transform 0.18s, box-shadow 0.18s;
}
.stroke-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    border-color: #c8c4b4;
}

/* Buttons */
button.primary {
    font-family:'Inter',sans-serif !important; font-weight:500 !important;
    background:#1a1a1a !important; color:#f4f1ea !important;
    border:none !important; border-radius:50px !important;
    padding:0.65rem 1.8rem !important; transition:all 0.2s !important;
    letter-spacing:0.03em !important;
}
button.primary:hover { background:#404040 !important; transform:translateY(-1px) !important; }
button.secondary {
    font-family:'Inter',sans-serif !important;
    background:transparent !important; color:#5a5550 !important;
    border:1px solid #ccc8ba !important; border-radius:50px !important;
    padding:0.55rem 1.4rem !important;
}
button.secondary:hover { background:#f0ece3 !important; }
button.stop {
    background:transparent !important; color:#c0392b !important;
    border:1px solid #e0beba !important; border-radius:50px !important;
    font-family:'Inter',sans-serif !important;
}

/* Labels */
label span, .gradio-container label {
    font-family:'Inter',sans-serif !important; font-size:0.78rem !important;
    color:#9b9080 !important; letter-spacing:0.04em !important;
}

/* Textarea / inputs */
textarea, select, .gradio-container input[type=text] {
    background:#fffef9 !important; border:1px solid #e0dcd0 !important;
    border-radius:6px !important; color:#2d3436 !important;
    font-family:'Inter',sans-serif !important;
}

/* Markdown */
.gradio-container h2 { font-family:'EB Garamond',serif !important;
                        font-weight:400 !important; color:#1a1a1a !important; }
.gradio-container h3 { font-family:'EB Garamond',serif !important;
                        font-weight:400 !important; color:#2d3436 !important; }
.gradio-container p, .gradio-container li { color:#5a5550 !important;
                                             line-height:1.7 !important; }
table { width:100%; border-collapse:collapse; }
th { background:#f0ece3; color:#5a5550; font-size:0.75rem; letter-spacing:0.08em;
     text-align:left; padding:7px 12px; border-bottom:1px solid #e0dcd0;
     text-transform:uppercase; font-family:'Inter',sans-serif; }
td { padding:7px 12px; border-bottom:1px solid #ece8e0; color:#2d3436; font-size:0.87rem; }
code { font-family:'JetBrains Mono',monospace; background:#f0ece3;
       color:#5a5550; padding:1px 5px; border-radius:3px; font-size:0.85em; }
blockquote { border-left:3px solid #ccc8ba; padding:0.5rem 1rem;
             background:rgba(255,255,255,0.5); border-radius:0 6px 6px 0;
             margin:0.5rem 0; font-style:italic; color:#7a7060 !important; }

.info-panel {
    background:rgba(255,255,255,0.55); border:1px solid #e8e4da;
    border-radius:10px; padding:1.2rem 1.4rem; font-size:0.92rem;
    line-height:1.7; color:#5a5550;
}

footer { display:none !important; }
.gradio-container .block { background:transparent !important; border:none !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Digital Calligrapher") as demo:

    _state = gr.State(_initial_state())

    gr.HTML("""
    <div class="callig-header">
        <div class="callig-title">Digital Calligrapher</div>
        <div class="callig-sub">Learning Zen Aesthetics through Human Feedback</div>
        <div class="callig-badges">
            <span class="badge">Bradley-Terry RLHF</span>
            <span class="badge">Pairwise Preference Learning</span>
            <span class="badge">6-Dimensional Aesthetic Space</span>
            <span class="badge">Hill-Climb Optimisation</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — The Studio (pairwise voting)
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("✦ The Studio"):

            with gr.Row():
                # ── Left: comparison ──────────────────────────────────────
                with gr.Column(scale=3):
                    gr.HTML("""
                    <div style='text-align:center;margin-bottom:1.2rem;'>
                        <div style='font-family:"EB Garamond",serif;font-size:1.3rem;
                                    color:#1a1a1a;letter-spacing:0.05em;'>
                            Which stroke moves you?
                        </div>
                        <div style='font-size:0.85rem;color:#9b9080;margin-top:0.3rem;'>
                            Choose intuitively — there is no right answer.
                        </div>
                    </div>
                    """)

                    with gr.Row():
                        with gr.Column(elem_classes="stroke-card"):
                            img_a = gr.Image(show_label=False, interactive=False,
                                             height=280)
                            btn_a = gr.Button("Prefer this  ←", variant="primary")
                        with gr.Column(elem_classes="stroke-card"):
                            img_b = gr.Image(show_label=False, interactive=False,
                                             height=280)
                            btn_b = gr.Button("→  Prefer this", variant="primary")

                    with gr.Row():
                        btn_skip  = gr.Button("Neither — show new pair", variant="secondary")
                        btn_reset = gr.Button("Reset all preferences", variant="stop")

                    gr.HTML("<div style='height:0.5rem'></div>")

                    ink_selector = gr.Dropdown(
                        list(INK_STYLES.keys()),
                        value="Sumi Ink",
                        label="Ink style",
                        info="Changes the visual character of all strokes",
                    )

                # ── Right: live profile ───────────────────────────────────
                with gr.Column(scale=2):
                    gr.HTML("""
                    <div class='info-panel' style='margin-bottom:1rem;'>
                        <strong style='font-family:"EB Garamond",serif;font-size:1.05rem;'>
                            What is RLHF?
                        </strong><br><br>
                        Reinforcement Learning from Human Feedback teaches AI concepts
                        that cannot be mathematically defined — like elegance, tension,
                        or flow. Each vote you cast updates a reward model that maps
                        the 6-dimensional stroke parameter space to your personal taste.
                        <br><br>
                        After enough votes, the <em>Masterpiece</em> tab will generate
                        strokes that genuinely maximise your learned preferences.
                    </div>
                    """)
                    gr.HTML("<div style='font-family:\"EB Garamond\",serif;font-size:1rem;color:#5a5550;margin-bottom:0.5rem;font-style:italic;'>Artist Profile</div>")
                    profile_html = gr.HTML(_status_html(_initial_state()))

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Masterpiece
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("✦ Masterpiece"):

            gr.HTML("""
            <div style='text-align:center;margin-bottom:1.5rem;'>
                <div style='font-family:"EB Garamond",serif;font-size:1.3rem;color:#1a1a1a;'>
                    Your Aesthetic, Crystallised
                </div>
                <div style='font-size:0.85rem;color:#9b9080;margin-top:0.3rem;'>
                    The model performs hill-climbing optimisation to find
                    the stroke that maximises your learned reward function.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class='info-panel'>
                        <strong>How it works:</strong><br><br>
                        <ol style='padding-left:1.2rem;margin:0;'>
                            <li>600 random parameter candidates are generated</li>
                            <li>Each is scored by the learned reward model</li>
                            <li>The highest-scoring candidate is refined
                                via gradient ascent in parameter space</li>
                            <li>The resulting stroke is rendered in your chosen ink</li>
                        </ol>
                        <br>
                        <em>The more votes you cast, the more personalised the result.</em>
                    </div>
                    """)
                    btn_gen    = gr.Button("✦ Craft Masterpiece", variant="primary")
                    gen_status = gr.Markdown("*Cast votes in The Studio, then craft your masterpiece.*")

                with gr.Column(scale=2, elem_classes="stroke-card"):
                    masterpiece_img = gr.Image(show_label=False, interactive=False,
                                               height=380)

        # ══════════════════════════════════════════════════════════════════
        # Tab 3 — Analytics
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("✦ Analytics"):

            gr.HTML("""
            <div style='text-align:center;margin-bottom:1.2rem;'>
                <div style='font-family:"EB Garamond",serif;font-size:1.3rem;color:#1a1a1a;'>
                    Your Aesthetic Profile, Visualised
                </div>
                <div style='font-size:0.85rem;color:#9b9080;margin-top:0.3rem;'>
                    Click Refresh to pull the latest charts from your voting session.
                </div>
            </div>
            """)

            btn_refresh = gr.Button("⟳ Refresh Analytics", variant="secondary")

            with gr.Row():
                lc_chart  = gr.Image(label="Learning Curve", show_label=True,
                                     interactive=False)
                wb_chart  = gr.Image(label="Preference Weights", show_label=True,
                                     interactive=False)

            radar_chart = gr.Image(label="Aesthetic Radar", show_label=True,
                                   interactive=False)

            btn_refresh.click(cb_refresh_charts, [_state],
                              [lc_chart, radar_chart, wb_chart])

        # ══════════════════════════════════════════════════════════════════
        # Tab 4 — How RLHF Works
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("✦ How It Works"):

            gr.Markdown("""
## The Mathematics of Aesthetic Learning

This app is a miniature version of the same technique used to align large
language models like ChatGPT with human values. The core idea: instead of
defining "beautiful" mathematically, we let humans compare options and learn
from those comparisons.

---

## The Bradley-Terry Model

Given two strokes A and B, we model the probability that a human prefers A:

$$P(A \\succ B) = \\sigma(R(A) - R(B)) = \\frac{1}{1 + e^{-(R(A)-R(B))}}$$

where $R(x) = \\mathbf{w}^\\top \\mathbf{x}$ is a linear reward function and
$\\sigma$ is the sigmoid function.

**Training** minimises the cross-entropy loss over all comparisons:

$$\\mathcal{L} = -\\log P(A \\succ B) = \\log(1 + e^{-(R(A)-R(B))})$$

**Gradient update** (one step per vote):

$$\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta \\cdot \\sigma(-(R(A)-R(B))) \\cdot (\\mathbf{x}_A - \\mathbf{x}_B)$$

The update pushes weights toward features that distinguish preferred strokes.

---

## The 6 Stroke Parameters

| Parameter | Range | Effect |
|---|---|---|
| `Complexity` | 3–10 points | Number of control points — wilder or smoother path |
| `Smoothness` | degree 2–4 | Spline polynomial degree — how tightly path follows points |
| `Pressure` | 1×–5.5× | Peak line width — delicate hairline to bold brush |
| `Curvature` | 0–2.2 | Sine wave amplitude applied to y-coordinates |
| `Length` | 0.3–1.0 | Horizontal span of the stroke |
| `Randomness` | 0–0.15 | Jitter added to control points — organic vs precise |

---

## Why Pairwise Comparisons?

Absolute ratings are unreliable — "7 out of 10" means different things to different
people and changes over time. Pairwise comparisons ("I prefer A over B") are much more
consistent and map directly onto the Bradley-Terry likelihood.

This is why RLHF systems like InstructGPT, Claude, and Gemini use human annotators
who compare model outputs rather than rate them absolutely.

---

## Hill-Climb Optimisation

Once preferences are learned, generating a "masterpiece" requires finding
$\\mathbf{x}^* = \\arg\\max_{\\mathbf{x} \\in [0,1]^6} \\mathbf{w}^\\top \\mathbf{x}$.

For a linear reward, the analytical solution is:
$x_i^* = 1$ if $w_i > 0$, else $x_i^* = 0$.

But this produces degenerate strokes (all parameters at extremes). Instead we:
1. Sample 600 random candidates
2. Pick the highest-scoring
3. Refine with gradient ascent (small steps in the direction of **w**)
4. This finds good local optima with natural parameter values

---

## Connection to Full RLHF

| This app | Real RLHF (e.g. InstructGPT) |
|---|---|
| 6-dim linear reward model | Deep neural network reward model |
| Pairwise stroke votes | Pairwise LLM output rankings by annotators |
| Gradient update per vote | Batched gradient descent |
| Hill-climb parameter search | PPO policy optimisation |
| Immediate personalisation | Generalised alignment |
""")

    gr.HTML("""
    <div style='text-align:center;font-family:"EB Garamond",serif;font-style:italic;
                font-size:0.9rem;color:#c8c4b4;padding:2.5rem 0 1rem;
                border-top:1px solid #e8e4da;margin-top:1rem;'>
        Drawing with intention — guided by your aesthetic, shaped by your choices.
    </div>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────

    demo.load(cb_init, [_state], [img_a, img_b, profile_html, _state])

    btn_a.click(lambda s: cb_vote("A", s), [_state],
                [img_a, img_b, profile_html, _state])
    btn_b.click(lambda s: cb_vote("B", s), [_state],
                [img_a, img_b, profile_html, _state])
    btn_skip.click(cb_skip, [_state], [img_a, img_b, _state])

    ink_selector.change(cb_change_ink, [ink_selector, _state],
                        [img_a, img_b, _state])

    btn_gen.click(cb_generate_masterpiece, [_state],
                  [masterpiece_img, gen_status, _state])

    btn_reset.click(cb_reset, [_state],
                    [img_a, img_b, profile_html,
                     lc_chart, radar_chart, wb_chart, _state])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
