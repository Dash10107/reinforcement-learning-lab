---
title: Digital Calligrapher Rlhf
emoji: 🏆
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# Digital Calligrapher — Learning Aesthetics with RLHF

An interactive playground that teaches an AI your personal sense of beauty through pairwise comparisons. You look at two calligraphic brush strokes and pick the one that feels more elegant to you. Behind the scenes, a reward model updates its understanding of what you find aesthetically pleasing. After enough votes, the system can generate strokes that are genuinely optimised for your taste.

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Dash10107/digital-calligrapher-rlhf)

---

## What problem does this solve?

"Elegance" cannot be written down as a formula. You cannot define it mathematically, hand it to an optimiser, and get beautiful output. But humans can compare two options and say which one they prefer. RLHF — Reinforcement Learning from Human Feedback — turns those comparisons into a learnable signal.

This is the same core technique used to align large language models like ChatGPT and Claude with human values. Here we apply it to something more visual and tangible: teaching an AI what makes a brush stroke beautiful, according to you specifically.

---

## The Algorithm: RLHF with Bradley-Terry Reward Modelling

Traditional RL requires a reward function defined upfront. RLHF removes that requirement. Instead of a predefined reward, humans provide relative comparisons, and the system learns a reward model from those comparisons.

**The Bradley-Terry model** is the statistical foundation. It says that given two options A and B, the probability that a person prefers A is:

```
P(A preferred over B) = sigmoid(R(A) - R(B))
```

where R is the reward model's score for each option. The sigmoid function turns the score difference into a probability between 0 and 1.

**Training:** Each time you vote, the model updates its weights using the gradient of the cross-entropy loss:

```
loss = -log( sigmoid(R(preferred) - R(rejected)) )

gradient = sigmoid(-(R(preferred) - R(rejected))) * (features_preferred - features_rejected)

weights = weights + learning_rate * gradient
```

The update pushes the weights to score the preferred stroke higher than the rejected one. Over many votes this shapes a weight vector that encodes your aesthetic preferences.

**The reward model** here is a simple linear model: six weights, one per stroke parameter. This keeps everything interpretable — you can look at the weights and see exactly what the model has learned about your taste.

---

## The Stroke Parameterisation

Each stroke is fully described by six parameters, all normalised between 0 and 1:

| Parameter | Effect on the stroke |
|---|---|
| Complexity | Number of control points — more points means a more tortuous path |
| Smoothness | Degree of the spline fit — higher means a more flowing, less angular curve |
| Pressure | Peak line width — ranges from a delicate hairline to a broad, bold stroke |
| Curvature | Amplitude of the sine wave applied to the vertical axis |
| Length | How far the stroke extends horizontally across the canvas |
| Randomness | Jitter added to control point positions — organic vs geometric feel |

When the model finishes voting, it has a 6-dimensional weight vector. The sign of each weight tells you whether you prefer more or less of that property. The magnitude tells you how strongly you feel about it.

---

## Generating a Masterpiece

Once preferences are learned, generating the optimal stroke requires finding the parameter vector that maximises the reward:

```
best_params = argmax over [0,1]^6 of (weights dot params)
```

For a linear reward model the analytical solution is just setting each parameter to 1 if its weight is positive and 0 if negative. But that produces extreme, unnatural strokes. Instead the app uses random-restart hill climbing:

1. Sample 600 random parameter vectors
2. Score each with the learned reward model
3. Take the highest-scoring candidate
4. Refine it by taking small steps in the direction of the weight vector for 200 iterations
5. Render the result

This finds a realistic optimum rather than a degenerate extreme.

---

## Project Structure

```
digital-calligrapher-rlhf/
├── app.py                  all code — environment, reward model, UI, analytics
└── requirements.txt
```

The project is deliberately kept as a single file. The reward model, stroke generator, visualisation, and Gradio UI all live in `app.py`. This makes it easy to read end-to-end without jumping between files.

---

## Quick Setup

Clone and install:

```bash
git clone https://github.com/yourusername/rl-portfolio
cd digital-calligrapher-rlhf
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Open `http://localhost:7860`.

**To try it out:** Go to The Studio tab. Two strokes will appear side by side. Pick the one that appeals more to you intuitively — there is no right answer. After around 10 to 15 votes you will start to see the Analytics tab show a clear preference profile. Then visit the Masterpiece tab to generate a stroke optimised for your aesthetic.

---

## What each tab shows

**The Studio:** Two strokes appear side by side. Vote with the two preference buttons or skip if you are genuinely indifferent. A live profile on the right updates after each vote, showing the six learned weights and a confidence score derived from how stable the loss has been recently. You can also change the ink style — seven options from Sumi Ink to Vermillion to Midnight.

**Masterpiece:** Runs the hill-climbing optimiser and renders the result. The status message shows the reward score the model assigned to the generated stroke. More votes = more personalised output.

**Analytics:** Three charts — a learning curve showing cross-entropy loss over votes, a radar chart showing your aesthetic profile across the six dimensions, and a horizontal bar chart showing raw weight values with positive and negative contributions clearly visible.

**How It Works:** Explains the Bradley-Terry model, the gradient update, the hill-climbing optimiser, and how each chart should be interpreted.

---

## Reading the Analytics

**Learning curve:** Cross-entropy loss should trend downward as the model becomes more consistent with your preferences. A flat or rising curve means your votes are contradictory — which is fine and reflects genuine ambiguity in aesthetic judgement.

**Radar chart:** Values normalised to 0-1. A spike toward Pressure means you prefer bold, wide strokes. A spike toward Curvature means you like arching, wave-like forms.

**Weight bar chart:** Green bars are properties you like more of. Red bars are properties you like less of. The magnitude shows how strongly you feel about each one.

---

## Requirements

```
numpy
matplotlib
scipy
gradio>=6.0.0
```

---

## Further reading

- Christiano et al., "Deep Reinforcement Learning from Human Preferences" (2017) — the foundational RLHF paper
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019) — applying RLHF to language
- Bradley and Terry, "Rank Analysis of Incomplete Block Designs" (1952) — the statistical model underpinning pairwise comparison learning

## Things to Try

**1. Vote consistently on one dimension and watch convergence.**
Always pick the stroke that looks more complex. After 15 votes open Analytics. The Complexity weight should be clearly positive and the radar chart should spike in that direction.

**2. Vote randomly for 10 rounds then consistently for 10.**
The learning curve should show a flat section followed by a downward trend. This demonstrates how the model handles contradictory data.

**3. Generate a masterpiece at different vote counts.**
Generate one after 5 votes, then 20, then 40. Save each. The strokes should become progressively more aligned with your preferences as the reward model gains confidence.

**4. Switch ink styles and regenerate.**
After training preferences, switch from Sumi Ink to Vermillion and regenerate. The stroke shape is identical — only colour and background change. Your trained preferences are ink-agnostic.

**5. Try to confuse the model.**
Vote for thin delicate strokes for 10 rounds, then deliberately vote for thick bold strokes for 10 rounds. Check the Pressure weight in Analytics — it should be near zero, reflecting genuine uncertainty.