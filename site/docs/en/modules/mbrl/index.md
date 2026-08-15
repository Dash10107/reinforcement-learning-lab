---
description: "Model-Based Reinforcement Learning (MBRL) explained. Covers world models, Model Predictive Control (MPC), Dyna-Q algorithm, ensemble uncertainty, and sample efficiency comparison."
---

# Model-Based RL Tutorial — Ensemble Models and MPC
<br> *What if the agent could imagine?*

Every algorithm in this course so far has been **model-free**.

The agent acts. The environment responds. The agent learns from what happened. But it has no model of the environment — no internal simulation it can run before acting. It can only learn from real experience.

Model-free methods work well. But real experience is expensive. Training an SAC agent to land a rocket requires thousands of actual landing attempts. Training a robot to walk requires hours of physical simulation. In the real world, this is costly, slow, and sometimes dangerous.

What if the agent could *imagine* what would happen, without actually doing it?

That's the idea behind model-based RL.

---

## The world model

A **world model** is a learned simulation: a neural network that takes a state and an action, and predicts the next state and the reward.

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                     nn.ReLU(),
            nn.Linear(256, state_dim + 1),           # predict next_state + reward
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        next_state = out[..., :-1]
        reward     = out[..., -1:]
        return next_state, reward
```

Once trained, the world model lets the agent "experience" trajectories inside its own head. Instead of taking 1,000 real steps in the environment, it takes 10 real steps and 990 imagined steps. Real experience trains the world model; the world model generates training data for the policy.

---

## Model Predictive Control (MPC): plan every step

The simplest way to use a world model is **MPC**: at each step, use the world model to simulate many possible futures, pick the sequence of actions that leads to the best outcome, execute only the first action, then re-plan.

```python
def mpc_action(world_model, state, n_candidates=100, horizon=10):
    # Generate random candidate action sequences
    candidates = torch.randn(n_candidates, horizon, action_dim)

    # Simulate each candidate in the world model
    best_return, best_first_action = float('-inf'), None

    for i in range(n_candidates):
        s = state.clone()
        total_reward = 0
        for t in range(horizon):
            action = candidates[i, t]
            s, r = world_model(s, action)
            total_reward += (gamma ** t) * r

        if total_reward > best_return:
            best_return = total_reward
            best_first_action = candidates[i, 0]

    return best_first_action   # only execute the first step; re-plan next step
```

Notice: MPC doesn't learn a policy in the traditional sense. It plans from scratch at every single step. This is slow at runtime but extremely flexible — it can adapt immediately to changes in the environment.

---

## Dyna-Q: the classic model-based approach

Before neural world models, **Dyna-Q** (Sutton, 1990) was the canonical model-based RL algorithm. It's elegant in its simplicity:

1. Take a real step in the environment. Update Q-values as usual (real experience).
2. Update the model: store the transition $(s, a) \rightarrow (r, s')$.
3. Do $n$ imagined steps: sample random past $(s, a)$ pairs from memory, use the model to predict $(r, s')$, update Q-values again.

```python
class DynaQ:
    def __init__(self, n_states, n_actions, planning_steps=10):
        self.Q     = np.zeros((n_states, n_actions))
        self.model = {}        # stores: (state, action) -> (reward, next_state)
        self.n     = planning_steps

    def update(self, s, a, r, s_next, alpha=0.1, gamma=0.99):
        # 1. Real Q-learning update
        self.Q[s, a] += alpha * (r + gamma * np.max(self.Q[s_next]) - self.Q[s, a])

        # 2. Model update: remember what happened
        self.model[(s, a)] = (r, s_next)

        # 3. Planning: learn from imagined experience
        for _ in range(self.n):
            # Sample a random past (state, action) pair
            s_sim, a_sim = random.choice(list(self.model.keys()))
            r_sim, s_sim_next = self.model[(s_sim, a_sim)]
            # Update Q-value as if this transition just happened
            self.Q[s_sim, a_sim] += alpha * (
                r_sim + gamma * np.max(self.Q[s_sim_next]) - self.Q[s_sim, a_sim]
            )
```

With `planning_steps = 50`, the agent effectively does 50 Q-learning updates per real environment step. In sparse-reward mazes, this makes Dyna-Q converge 50× faster than pure Q-learning. The model here is just a dictionary — it memorises transitions exactly. Neural world models are the continuous-state generalisation of this idea.

---

## Ensemble uncertainty: knowing when not to trust the model

A single world model doesn't know what it doesn't know. If the agent visits a state it's never seen before, the model will still output a prediction — with no indication that this prediction might be wrong.

The solution: train an **ensemble** of $N$ world models on the same data.

```python
class EnsembleWorldModel:
    def __init__(self, n_models=5, state_dim, action_dim):
        self.models = [WorldModel(state_dim, action_dim) for _ in range(n_models)]

    def predict(self, state, action):
        predictions = [m(state, action) for m in self.models]
        next_states  = torch.stack([p[0] for p in predictions])
        rewards      = torch.stack([p[1] for p in predictions])

        mean_state = next_states.mean(0)
        uncertainty = next_states.std(0).mean()   # disagreement = uncertainty
        mean_reward = rewards.mean(0)
        return mean_state, mean_reward, uncertainty
```

When all 5 models agree on the next state, uncertainty is low — the model is reliable. When they disagree, uncertainty is high — the agent should be cautious, or go explore that region to get more real data.

This ensemble uncertainty is the core mechanism in **MBPO** (Model-Based Policy Optimization, 2019) — the modern neural MBRL algorithm that achieves near-SAC performance with 10× fewer real environment steps.

---

## The tradeoff: real vs imagined experience

Model-based methods are more **sample efficient**: they learn faster from fewer real interactions. This is critical when real experience is costly (physical robots, real-world systems).

But they have a weakness: **model error**. The world model is always slightly wrong. The real pendulum has slightly different friction than the model predicts. The real wind gusts more than the simulation expects.

If the agent trains heavily on the model, it can optimise against the model's errors — a phenomenon called **model exploitation**. It finds actions that look great in the model but fail in reality.

The fix is to balance real and imagined experience carefully, and to quantify how uncertain the model is — using the uncertainty to avoid exploring regions where the model is unreliable.

---

## The sim-to-real gap

This is a hard and unsolved problem in robotics. Agents trained entirely in simulation often fail when deployed on physical hardware — not because they're bad agents, but because the simulation wasn't accurate enough.

The model-based RL project in this course uses the Pendulum environment: a classic control problem where the agent must swing a pendulum upright and keep it balanced. It's a good test case for MBRL because:

- The dynamics are complex enough that a model needs to be learned, not hand-coded
- But the environment is simple enough that we can clearly see model error and how it compounds

---

## What you'll notice in the demo

Open the [Pendulum MBRL ↗](https://huggingface.co/spaces/Dash10107/mbrl-pendulum-playground) — an agent balancing a pendulum using a learned world model.

**Three things to watch:**

1. **World model training.** Watch the predicted next-state vs actual next-state. Early: wild divergence. After 200 steps: they track closely. The model has learned pendulum physics from scratch.

2. **MPC horizon effects.** A longer planning horizon lets the agent look further ahead — but model errors compound over time. At horizon 1, it's reactive. At horizon 20, it's smooth but sometimes overconfident.

3. **Model exploitation.** Push the model to plan 50 steps ahead. Watch it start taking unusual actions that look optimal in the model but cause erratic real-world behaviour. That's model exploitation in action.

---

## Try it yourself

**Experiment 1 — Model-free baseline.**
Switch to a model-free SAC agent on the same task. Compare sample efficiency: SAC needs far more real environment steps to reach the same performance as MBRL. This is the core advantage of having a world model.

**Experiment 2 — Noisy model.**
Add Gaussian noise to the world model's predictions. Watch how the agent's performance degrades as model error increases. There's a noise threshold below which MPC still works well — beyond it, the plans become useless.

**Experiment 3 — Short horizon vs long horizon.**
Compare MPC at horizon 1, 5, and 15. Horizon 1 is myopic — the agent only thinks one step ahead. Horizon 15 plans further but makes stronger assumptions about the model's accuracy. Find the sweet spot for the pendulum.

---

## Next

We've gone from simple bandits to multi-agent swarms to agents that imagine the future. One question has been lurking beneath all of it: **what does the reward function reward?**

Who decides what the agent should care about? Who writes the reward function?

For simple games, we know the reward: win the game. But for complex, open-ended tasks — write a good essay, make a compelling argument, draw something beautiful — the reward isn't obvious. And if we get it wrong, the agent optimises for the wrong thing.

The next chapter is about how we solve this with human feedback.
