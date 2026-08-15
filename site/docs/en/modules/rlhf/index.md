---
description: "RLHF (Reinforcement Learning from Human Feedback) explained — the technique behind ChatGPT and Claude. Covers Bradley-Terry preference model, reward model training, KL penalty, and DPO."
---

# Teaching AI What You Actually Want

You can't write down a reward function for "this is beautiful."

You know it when you see it. But translating that into a number — a reward signal an RL agent can learn from — is nearly impossible. Any reward you write will be imperfect, and the agent will find a way to maximise it that isn't what you meant.

We saw a small version of this problem earlier with reward hacking — the boat racing agent that circled for coins instead of finishing the race.

Now imagine that problem at the scale of a language model. You want the AI to be helpful, honest, and harmless. How do you write a reward function for that?

RLHF — Reinforcement Learning from Human Feedback — is the answer. And it's running inside ChatGPT right now.

---

## The three steps of RLHF

**Step 1: Collect human preferences.**

Instead of defining what's good with a number, you show a human two outputs and ask: "Which is better?"

Two calligraphy strokes. Two essay responses. Two robot movements. The human picks the one they prefer. You collect thousands of these comparisons.

**Step 2: Train a reward model.**

You use those preference pairs to train a neural network — the **reward model** — that learns to predict which outputs humans prefer.

```python
class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 1),         # scalar reward
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def preference_loss(reward_model, chosen, rejected):
    r_chosen   = reward_model(chosen)
    r_rejected = reward_model(rejected)
    # The chosen output should score higher than the rejected one
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

The reward model learns the latent human preference function — the thing that was in the human's head when they compared the two outputs — from pairs of examples.

The training objective comes from the **Bradley-Terry preference model**. Given a pair of outputs $(y_w, y_l)$ where $y_w$ is preferred over $y_l$:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]$$

Where:
- $r_\phi(x, y)$ = the reward model's score for output $y$ given input $x$
- $\sigma(\cdot)$ = the sigmoid function — maps any real number to $(0,1)$
- $r_\phi(x, y_w) - r_\phi(x, y_l)$ = how much higher the preferred output scores
- $\log \sigma(\cdot)$ = log-probability that the model correctly predicts the preference

In plain English: *train the reward model to give higher scores to preferred outputs. If the score gap is large (the model is confident), the loss is small. If the model scores both outputs equally, the loss is maximised.*

**Step 3: Use PPO to optimise the reward model.**

Now you have a differentiable reward signal. Train the main agent (or language model) using PPO, with the reward model's score as the reward.

```python
# Training loop (simplified)
for batch in experience:
    # Get the reward model's score for each output
    reward = reward_model(batch.outputs).detach()

    # Use PPO to push the policy toward higher-scoring outputs
    ppo_update(policy, batch, reward)
```

The agent learns to produce outputs that the reward model would score highly — which means outputs that look like what humans preferred in the training pairs.

---

## Why this is profound

Before RLHF, aligning an AI system meant:
- Writing down rules explicitly (brittle, misses edge cases)
- Hoping the training objective captured what you actually wanted (it rarely did completely)

RLHF is different. It lets humans teach the reward function *through behaviour* — by comparing outputs — rather than by specifying it explicitly. Humans are much better at comparing two options than at writing down exactly what they want in advance.

This is why RLHF is the technique behind InstructGPT, Claude, Llama 2 — essentially every major language model deployed today. The task is different (generating text vs. drawing calligraphy), but the three steps are identical.

---

## KL divergence penalty: keeping it honest

A critical detail in RLHF for language models: after training the reward model, the PPO objective includes a **KL divergence penalty** against the original (pre-RLHF) model:

$$\text{Reward} = r_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)]$$

Where:
- $r_\phi(x, y)$ = reward model's score for response $y$
- $\pi_{\text{ref}}$ = the reference model (original, before RLHF fine-tuning)
- $\text{KL}[\pi_\theta \| \pi_{\text{ref}}]$ = how different the fine-tuned model is from the original
- $\beta$ = controls how strongly we penalise deviation from the reference

In plain English: *don't let the model drift too far from what it was before RLHF. The reward model is imperfect — if we maximise it unconstrained, the model will exploit its weaknesses.*

Without the KL penalty, the model finds responses that score extremely high on the reward model but are gibberish — a form of reward hacking at the language model level. The KL penalty keeps the model anchored to sensible language.

---

## DPO: skipping the reward model entirely

A newer approach, **Direct Preference Optimization (DPO)**, eliminates the separate reward model training step. Instead of:

> collect preferences → train reward model → run PPO

DPO directly optimises the policy on preference pairs:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

In plain English: *make the model more likely to produce preferred outputs and less likely to produce rejected ones — relative to the reference model — without ever training a separate reward model.*

DPO is simpler (one training phase instead of three), more stable (no RL instabilities), and achieves comparable results to RLHF in many benchmarks. Most modern open-source model alignment (Llama 3, Mistral fine-tunes) uses DPO or variants.

---

## The calligraphy project

We test RLHF on a deliberately subjective task: digital calligraphy.

An agent controls a virtual brush. It learns to produce strokes. Early in training, the strokes are random scribbles. A human labels pairs of strokes as "better" or "worse" based on their personal aesthetic sense. The reward model learns that preference. The agent trains to produce strokes the reward model would approve of.

What's beautiful about this setup: **there's no ground truth**. One person's beautiful stroke is another's mess. The agent learns *your* preference specifically — it personalises to the labeller.

This is exactly how RLHF in language models works. Different annotators emphasise different qualities (accuracy, tone, conciseness). The reward model captures a mix of those preferences. The final model reflects the specific labelling team's values.

---

## What you'll notice in the demo

Open the [Digital Calligrapher ↗](https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf) — you label pairs of strokes, and the agent learns your aesthetic.

**Three things to watch:**

1. **Pre-training.** Before you label anything, the strokes are random. The agent has a policy but no useful reward signal.

2. **After 20 labels.** The reward model has enough signal to distinguish good from bad. Watch the strokes start to drift toward what you preferred. They're not perfect yet, but the direction is right.

3. **Reward model uncertainty.** When two strokes look similar to you, the reward model is uncertain — it gives them similar scores. When they're clearly different, it's confident. You can see this in the score gap between preferred and rejected outputs.

---

## Try it yourself

**Experiment 1 — Contradictory preferences.**
Label the first 10 pairs one way (prefer thin strokes). Label the next 10 the opposite way (prefer thick strokes). Watch the reward model become confused — its loss plateaus and the agent oscillates. Consistency in labelling matters.

**Experiment 2 — Minimal labels.**
Train the reward model with only 5 pairs. The agent often finds degenerate solutions that score well in the reward model but look wrong to a human. This is reward model overfitting — the reward model hasn't seen enough diversity.

**Experiment 3 — Two labellers.**
Have two people label the same pairs independently. Compare their preference labels. Notice where they disagree. Now ask: whose values should the model learn? This is not a technical question. It's the core alignment question.

---

## The bigger picture

RLHF is not a complete solution to AI alignment. It's a step.

The reward model is only as good as the labellers. The labellers can be wrong, inconsistent, or biased. The agent can find ways to score highly on the reward model that don't correspond to truly good outputs — a form of reward hacking at the RLHF level.

But it's a better step than anything that came before. Teaching preference through comparison — rather than specification through rules — is more natural and more scalable. It's the direction the field is moving.
