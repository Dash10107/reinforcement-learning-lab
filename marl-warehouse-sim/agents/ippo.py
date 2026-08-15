"""
Independent PPO for the WarehouseEnv.
Each robot has its own Actor-Critic pair (no parameter sharing).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from warehouse.env import N_ACTIONS, WarehouseEnv

from agents.networks import Actor, Critic

CHECKPOINT = "warehouse_ippo.pt"


@dataclass
class TrainingState:
    running: bool = False
    episode: int = 0
    total_episodes: int = 0
    mean_rewards: list[float] = field(default_factory=list)
    deliveries_per_ep: list[float] = field(default_factory=list)
    collisions_per_ep: list[float] = field(default_factory=list)
    status: str = "No agents trained yet."
    model_ready: bool = False
    _recent: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def rolling_mean(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0


class RobotAgent:
    def __init__(self, obs_dim: int, lr: float = 3e-4):
        self.actor = Actor(obs_dim, N_ACTIONS)
        self.critic = Critic(obs_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr * 2)

        self.obs_buf, self.act_buf, self.lp_buf = [], [], []
        self.rew_buf, self.val_buf, self.done_buf = [], [], []

    def act(self, obs: np.ndarray):
        t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            a, lp = self.actor.act(t)
            v = self.critic(t)
        return int(a.item()), float(lp.item()), float(v.item())

    def store(self, obs, action, lp, reward, done, value):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.lp_buf.append(lp)
        self.rew_buf.append(reward)
        self.done_buf.append(done)
        self.val_buf.append(value)

    def update(
        self,
        last_obs: np.ndarray,
        gamma: float = 0.95,
        lam: float = 0.95,
        clip: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        ent_coef: float = 0.01,
    ) -> dict:

        with torch.no_grad():
            last_v = float(self.critic(torch.FloatTensor(last_obs).unsqueeze(0)).item())

        # GAE
        adv = np.zeros(len(self.rew_buf), dtype=np.float32)
        gae = 0.0
        vals = np.array(self.val_buf + [last_v], dtype=np.float32)
        for t in reversed(range(len(self.rew_buf))):
            delta = (
                self.rew_buf[t] + gamma * vals[t + 1] * (1 - self.done_buf[t]) - vals[t]
            )
            gae = delta + gamma * lam * (1 - self.done_buf[t]) * gae
            adv[t] = gae
        ret = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.FloatTensor(np.array(self.obs_buf))
        act_t = torch.LongTensor(self.act_buf)
        olp_t = torch.FloatTensor(self.lp_buf)
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(ret)

        al_total = cl_total = n = 0
        for _ in range(n_epochs):
            idx = np.random.permutation(len(obs_t))
            for s in range(0, len(idx), batch_size):
                bi = idx[s : s + batch_size]
                lp, ent = self.actor.evaluate(obs_t[bi], act_t[bi])
                ratio = (lp - olp_t[bi]).exp()
                s1 = ratio * adv_t[bi]
                s2 = ratio.clamp(1 - clip, 1 + clip) * adv_t[bi]
                al = -torch.min(s1, s2).mean() - ent_coef * ent.mean()
                cl = nn.functional.mse_loss(self.critic(obs_t[bi]), ret_t[bi])

                self.actor_opt.zero_grad()
                al.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.zero_grad()
                cl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                al_total += al.item()
                cl_total += cl.item()
                n += 1

        self.obs_buf.clear()
        self.act_buf.clear()
        self.lp_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()
        self.done_buf.clear()
        return {"actor_loss": al_total / max(n, 1), "critic_loss": cl_total / max(n, 1)}


class WarehouseIPPO:
    def __init__(self, n_robots: int, obs_dim: int, lr: float = 3e-4):
        self.agents = {f"robot_{i}": RobotAgent(obs_dim, lr) for i in range(n_robots)}

    def act_all(self, obs_dict: dict) -> tuple[dict, dict, dict]:
        actions, lps, vals = {}, {}, {}
        for aid, agent in self.agents.items():
            obs = obs_dict[aid]
            a, lp, v = agent.act(obs)
            actions[aid] = a
            lps[aid] = lp
            vals[aid] = v
        return actions, lps, vals

    def greedy_all(self, obs_dict: dict) -> dict:
        return {
            aid: agent.actor.greedy(obs_dict[aid]) for aid, agent in self.agents.items()
        }

    def store_all(self, obs, actions, lps, rewards, dones, vals):
        for aid, agent in self.agents.items():
            agent.store(
                obs[aid], actions[aid], lps[aid], rewards[aid], dones[aid], vals[aid]
            )

    def update_all(self, last_obs: dict) -> dict:
        return {aid: agent.update(last_obs[aid]) for aid, agent in self.agents.items()}

    def save(self, path: str = CHECKPOINT):
        torch.save(
            {
                aid: {
                    "actor": a.actor.state_dict(),
                    "critic": a.critic.state_dict(),
                }
                for aid, a in self.agents.items()
            },
            path,
        )

    def load(self, path: str = CHECKPOINT) -> bool:
        if not os.path.exists(path):
            return False
        sd = torch.load(path, map_location="cpu")
        for aid, agent in self.agents.items():
            if aid in sd:
                agent.actor.load_state_dict(sd[aid]["actor"])
                agent.critic.load_state_dict(sd[aid]["critic"])
        return True


def start_training(
    n_robots: int,
    total_episodes: int,
    rollout_steps: int,
    state: TrainingState,
    lr: float = 3e-4,
) -> threading.Thread:

    def _run():
        state.running = True
        state.total_episodes = total_episodes
        env = WarehouseEnv(n_robots=n_robots, max_steps=rollout_steps, seed=0)
        ippo = WarehouseIPPO(n_robots, env.obs_dim, lr=lr)
        obs = env.reset()
        ep_rewards = {aid: 0.0 for aid in env.agent_ids}
        ep = 0

        while ep < total_episodes and state.running:
            # Collect rollout
            for _ in range(rollout_steps):
                actions, lps, vals = ippo.act_all(obs)
                next_obs, rewards, done, _info = env.step(
                    [actions[aid] for aid in env.agent_ids]
                )
                reward_dict = {aid: rewards[i] for i, aid in enumerate(env.agent_ids)}
                done_dict = {aid: done for aid in env.agent_ids}
                for aid in env.agent_ids:
                    ep_rewards[aid] += reward_dict[aid]
                ippo.store_all(obs, actions, lps, reward_dict, done_dict, vals)
                obs = next_obs
                if done:
                    state.deliveries_per_ep.append(env.total_deliveries)
                    state.collisions_per_ep.append(env.total_collisions)
                    mean_r = np.mean(list(ep_rewards.values()))
                    state.mean_rewards.append(mean_r)
                    state._recent.append(mean_r)
                    ep += 1
                    state.episode = ep
                    state.status = (
                        f"Ep {ep}/{total_episodes}  |  "
                        f"Reward: {mean_r:+.2f}  |  "
                        f"Deliveries: {env.total_deliveries}  |  "
                        f"Collisions: {env.total_collisions}  |  "
                        f"Rolling: {state.rolling_mean:+.2f}"
                    )
                    ep_rewards = {aid: 0.0 for aid in env.agent_ids}
                    obs = env.reset()
                    if ep >= total_episodes:
                        break

            ippo.update_all(obs)

        ippo.save(CHECKPOINT)
        state.model_ready = True
        state.running = False
        state.status = (
            f"Training complete!  {ep} episodes  |  "
            f"Best rolling: {state.rolling_mean:+.2f}  |  "
            f"Avg deliveries: {np.mean(state.deliveries_per_ep):.1f}/ep"
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
