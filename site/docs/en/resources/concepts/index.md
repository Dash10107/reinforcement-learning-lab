# Core Concepts Glossary

Every term used in this course, in the order they become relevant. One plain-English sentence each. Look here when something in a chapter doesn't click.

---

## The MDP Framework

**Agent** — the thing that makes decisions. In RL, this is your algorithm and its parameters.

**Environment** — everything the agent interacts with. Changes in response to the agent's actions.

**State (s)** — a description of the current situation. The agent makes decisions based on the state.

**Action (a)** — something the agent can do. Could be "move left", or "apply 12.7 Nm of torque to joint 3".

**Reward (r)** — a scalar number the environment gives the agent after each step. Positive means the agent did something good; negative means it did something bad.

**Policy (π)** — the agent's decision-making rule. Maps states to actions, or to probabilities over actions.

**Optimal policy (π*)** — the policy that maximises expected total reward. What every RL algorithm is trying to find.

**Episode** — one complete run of the agent in the environment, from start to finish.

**Discount factor (γ, gamma)** — a number between 0 and 1 that controls how much the agent cares about future rewards vs immediate ones. Common value: 0.99.

**Return (G)** — the total discounted reward accumulated from some point in time onward: $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$.

**Markov property** — the future only depends on the current state, not the full history of how you got there.

**MDP (Markov Decision Process)** — the formal framework describing any RL problem: (S, A, P, R, γ).

**Transition model (P)** — the probability of moving to state s' when you take action a in state s. Usually unknown; the agent has to discover it by trying.

---

## Value Functions

**Value function V^π(s)** — expected total reward from state s, following policy π from then on.

**Action-value function Q^π(s,a)** — expected total reward from state s when you take action a first, then follow policy π.

**Optimal Q-function Q*(s,a)** — the Q-function for the optimal policy. Finding this is equivalent to finding the optimal policy.

**Advantage function A(s,a)** — how much better action a is than the average action in state s: $A(s,a) = Q(s,a) - V(s)$.

**Bellman equation** — a recursive equation saying: "the value now equals the immediate reward plus the discounted value of the next state."

**Bellman optimality** — the same idea, but taking the maximum over actions: the optimal value of a state is the value of the best action from there.

---

## Exploration

**Explore-exploit tradeoff** — the fundamental tension: should the agent try new things (explore) or use what it already knows works (exploit)?

**Epsilon-greedy (ε-greedy)** — the simplest exploration strategy: with probability ε, take a random action; otherwise, take the best known action.

**Epsilon decay** — gradually reducing ε over training, so the agent explores more early and exploits more later.

**UCB (Upper Confidence Bound)** — an exploration strategy that adds an uncertainty bonus to each action's estimated value, guiding exploration toward uncertain actions.

**Thompson Sampling** — an exploration strategy that maintains a probability distribution over each action's true value and samples from it to decide what to try.

**Regret** — the total reward lost compared to always picking the best action from the start. Lower regret = smarter exploration.

---

## Tabular RL

**Q-table** — a lookup table storing a Q-value for every (state, action) pair. Only practical for small, discrete state spaces.

**Q-Learning** — an off-policy algorithm that updates Q-values using the best possible next action: $Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s',a')$.

**SARSA** — an on-policy algorithm that updates Q-values using the *actual* next action taken, not the theoretical best one.

**On-policy** — learning about the policy that's currently being used to collect data.

**Off-policy** — learning about a different (usually better) policy from data collected by the current one.

**Monte Carlo** — a family of methods that wait until the end of an episode before updating, using real returns rather than estimates.

---

## Temporal Difference

**TD error (δ)** — how wrong the current value estimate was: $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$.

**Bootstrapping** — using your own current estimates as targets for your own updates. Q-Learning bootstraps; Monte Carlo doesn't.

**n-step return** — using the actual rewards for n steps, then bootstrapping from the value estimate at step n.

**TD(λ)** — a weighted combination of all n-step returns, controlled by λ ∈ [0,1]. At λ=0: TD(0). At λ=1: Monte Carlo.

**Eligibility traces** — per-state counters tracking how recently a state was visited. Used to implement TD(λ) efficiently.

**Bias** — systematic error in an estimate. Bootstrapping from wrong estimates introduces bias.

**Variance** — random variation in an estimate. Full-episode returns have high variance because one lucky/unlucky episode can mislead the update.

---

## Deep RL

**Function approximation** — using a neural network (or other function) instead of a lookup table to represent V or Q.

**DQN (Deep Q-Network)** — Q-Learning with a neural network. Requires experience replay and a target network to train stably.

**Experience replay** — storing past experiences in a buffer and sampling random batches to train on. Breaks correlation between consecutive samples.

**Replay buffer** — the memory that stores (s, a, r, s') tuples for experience replay. Typically a fixed-size circular buffer.

**Target network** — a frozen copy of the online network used to compute stable training targets. Updated every N steps.

**Overestimation bias** — the tendency of Q-Learning to overestimate Q-values, because `max Q(s')` tends to pick lucky overestimates.

---

## Policy Gradient Methods

**Policy gradient** — a family of methods that directly optimise the parameters of a policy by following the gradient of expected reward.

**Policy Gradient Theorem** — proves that $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$.

**REINFORCE** — the simplest policy gradient algorithm: collect a full episode, compute returns, update the policy.

**Baseline** — a function subtracted from the return in a policy gradient update to reduce variance without changing the expected gradient.

**Entropy** — a measure of how random a policy is. High entropy = explores broadly; low entropy = committed to specific actions.

---

## Actor-Critic Methods

**Actor** — the policy network. Decides what action to take.

**Critic** — the value network. Estimates how good the current state is.

**A2C (Advantage Actor-Critic)** — synchronous Actor-Critic using the advantage function as the policy gradient signal.

**PPO (Proximal Policy Optimization)** — Actor-Critic with a clipping constraint that prevents large policy updates.

**PPO clipping** — limits the policy update ratio to [1-ε, 1+ε], preventing catastrophic policy changes.

**SAC (Soft Actor-Critic)** — Actor-Critic for continuous actions, with a maximum entropy objective and twin critics.

**Gaussian policy** — a policy that outputs the mean and standard deviation of a normal distribution, from which continuous actions are sampled.

**Polyak averaging** — smoothly updating a target network by blending: target = τ × online + (1-τ) × target. More stable than periodic hard copies.

---

## Multi-Agent RL

**MARL (Multi-Agent RL)** — RL with multiple agents that share an environment and potentially interact.

**Non-stationarity** — in MARL, each agent's effective environment changes as other agents learn. This breaks convergence guarantees for single-agent algorithms.

**IPPO (Independent PPO)** — each agent runs its own PPO algorithm independently, treating other agents as part of the environment.

**Cooperative setting** — agents share a reward or have aligned goals.

**Competitive setting** — agents have opposing goals; one agent's reward is another's loss.

**Emergent behaviour** — complex group behaviour that arises from simple individual policies without explicit coordination design.

---

## Model-Based RL

**Model-free** — learning from real environment interactions only, without an internal model of the world.

**Model-based** — learning a world model and using it for planning or generating imagined training data.

**World model** — a learned neural network that predicts the next state and reward given a state and action.

**MPC (Model Predictive Control)** — planning by simulating many candidate action sequences in the world model and executing only the first step.

**Sim-to-real gap** — the difference between how a world model (or simulator) behaves and how the real environment behaves.

**Model exploitation** — when an agent finds actions that score well in a learned model but fail in the real environment.

---

## RLHF

**RLHF (Reinforcement Learning from Human Feedback)** — learning a reward model from human preference comparisons, then using RL to optimise it.

**Reward model** — a neural network trained to predict which of two outputs a human would prefer. Used as the reward signal for RLHF.

**Preference pair** — two agent outputs shown to a human who picks which is better. The training data for the reward model.

**Reward hacking (RLHF)** — the agent finding outputs that score highly in the reward model but don't genuinely satisfy human preferences.

---

## Miscellaneous

**Curriculum learning** — starting with an easy version of a task and gradually increasing difficulty as the agent improves.

**Sparse reward** — an environment where the agent receives a non-zero reward only rarely (e.g., only upon success).

**Dense reward** — an environment where the agent receives informative reward signals at every or most steps.

**Credit assignment** — figuring out which past actions were responsible for a current outcome. One of the fundamental hard problems in RL.

**Sample efficiency** — how much real environment interaction an algorithm needs to learn a good policy. Model-based methods tend to be more sample-efficient than model-free methods.

**Gymnasium** — the standard Python library for RL environments (formerly OpenAI Gym). Provides a consistent interface that every algorithm in this course works with.

[← Back to Resources](../)
