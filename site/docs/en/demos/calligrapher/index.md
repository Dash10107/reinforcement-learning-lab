---
description: "Digital Calligrapher RLHF demo — label brush strokes as better or worse, train a reward model on your preferences, and watch PPO optimise for your personal aesthetic. Same technique as ChatGPT alignment."
---

# ✍️ Digital Calligrapher (RLHF)

You are the reward function.

An agent controls a virtual brush. It learns to produce strokes you find beautiful — not beautiful in any objective sense, but beautiful to *you*, based on your specific comparisons. This is Reinforcement Learning from Human Feedback, and it's the same technique that shaped ChatGPT's personality.

<a href="https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf" target="_blank" rel="noopener" class="hero-btn-primary" style="display:inline-flex;margin-bottom:1.5rem">
  ▶ Open Live Demo ↗
</a>

---

## What this demo shows

**The three-step RLHF loop.** The demo runs the complete RLHF pipeline:

1. **You label.** The demo shows you two brush strokes. You pick the better one. This is your preference signal.
2. **The reward model learns.** Behind the scenes, a neural network trains on your preference pairs. It learns to predict which strokes you'd prefer — the latent aesthetic model in your head.
3. **PPO optimises.** The brush agent trains against the reward model's scores. It learns to produce strokes the reward model (i.e., you) would approve of.

**Pre-training strokes.** Before you label anything, click "Generate stroke." The results are random — the policy is uniform, no preference signal. This is the baseline.

**After 20 labels.** The reward model has enough data to start differentiating. Watch the strokes start drifting toward what you preferred. They're not perfect, but the direction is right.

**The reward model's uncertainty.** When two strokes look similar to you, you'll hesitate — and the reward model does too. It gives them similar scores. When they're clearly different, confidence is high. Watch the score gap in the "reward model view" panel.

---

## Try these experiments

**Experiment 1 — Consistent preference.**
Pick a single aesthetic rule — "prefer thin strokes" — and label 30 pairs with strict consistency. Watch the agent converge to thin strokes. The more consistent your labels, the faster convergence.

**Experiment 2 — Contradictory labels.**
Label the first 10 pairs preferring thin strokes. Then label the next 10 preferring thick strokes. Watch the reward model's training loss plateau — it's confused. The agent oscillates. Consistency is the fuel for RLHF.

**Experiment 3 — Two aesthetic systems.**
Have two people label the same pairs independently. Compare their labels — notice where they disagree. Now: whose taste should the model learn? This is not a technical question. It's the core AI alignment question. Every RLHF deployment has to answer it.

**Experiment 4 — Reward model overfit.**
Label only 5 pairs and run full PPO training. Watch the agent find strokes that score very high on the reward model but look wrong to you. This is reward model overfitting — the agent has found a blind spot in a model that hasn't seen enough diversity. This is exactly the failure mode that the KL penalty in RLHF protects against.

---

## Why this is profound

Before RLHF, aligning an AI system required writing down explicit rules for what "good" means. But human preferences are complex, contextual, and often contradictory — we can't write them down.

What humans *can* do is compare two options and say which is better. We do this naturally, quickly, and consistently enough for a model to learn from.

RLHF captures this. It lets humans teach the reward function through behaviour rather than specification.

The same three steps — collect preferences, train reward model, run PPO — happen inside InstructGPT, Claude, and Llama 2. The task is different (text instead of brush strokes). The pipeline is identical.

---

## The chapter behind this demo

- **[Teaching AI What You Actually Want (RLHF)](../../modules/rlhf/)** — covers the Bradley-Terry preference model, the PPO training loop, the KL divergence penalty, and DPO as the modern alternative

---

**Difficulty:** Advanced · **Algorithms:** Reward Model + PPO · **Core concept:** RLHF / AI alignment
