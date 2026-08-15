"""
Independent PPO (IPPO) — each agent maintains its own Actor-Critic pair.
Agents are trained independently with shared architecture (parameter sharing optional).
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Callable

from agents.networks import Actor, Critic
from agents.buffer import Rollout, compute_gae, build_minibatches
from config import EnvConfig, PPOConfig, NetworkConfig


class IPPOAgent:
    def __init__(self, agent_id: str, env_cfg: EnvConfig, ppo_cfg: PPOConfig, net_cfg: NetworkConfig):
        self.agent_id = agent_id
        self.cfg = ppo_cfg
        self.actor = Actor(env_cfg.obs_dim, env_cfg.act_dim, net_cfg.hidden_sizes)
        self.critic = Critic(env_cfg.obs_dim, net_cfg.hidden_sizes)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ppo_cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=ppo_cfg.lr_critic)
        self.rollout = Rollout()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action, log_prob = self.actor.act(obs_t)
        value = self.critic(obs_t)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.actor(obs_t)
        return dist.probs.argmax().item()

    def store(self, obs, action, log_prob, reward, done, value):
        self.rollout.append(obs, action, log_prob, reward, done, value)

    def update(self, last_obs: np.ndarray) -> dict[str, float]:
        cfg = self.cfg
        with torch.no_grad():
            last_value = self.critic(torch.FloatTensor(last_obs).unsqueeze(0)).item()

        advantages, returns = compute_gae(
            self.rollout.rewards, self.rollout.values, self.rollout.dones,
            last_value, cfg.gamma, cfg.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.FloatTensor(np.array(self.rollout.obs))
        act_t = torch.LongTensor(self.rollout.actions)
        old_lp_t = torch.FloatTensor(self.rollout.log_probs)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)

        total_actor_loss = total_critic_loss = total_entropy = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in build_minibatches(
                obs_t, act_t, old_lp_t, adv_t, ret_t, cfg.batch_size
            ):
                new_lp, entropy = self.actor.evaluate(obs_b, act_b)
                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy.mean()

                values = self.critic(obs_b)
                critic_loss = nn.functional.mse_loss(values, ret_b)

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.rollout.clear()
        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


class IPPOTrainer:
    def __init__(self, env_cfg: EnvConfig, ppo_cfg: PPOConfig, net_cfg: NetworkConfig):
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg
        self.agents: dict[str, IPPOAgent] = {}
        self._net_cfg = net_cfg

    def init_agents(self, agent_ids: list[str]):
        for aid in agent_ids:
            self.agents[aid] = IPPOAgent(aid, self.env_cfg, self.ppo_cfg, self._net_cfg)

    def collect_rollout(self, env, steps: int) -> dict[str, float]:
        obs_dict, _ = env.reset()
        episode_rewards: dict[str, float] = {a: 0.0 for a in self.agents}

        for _ in range(steps):
            actions, log_probs, values = {}, {}, {}
            for aid, agent in self.agents.items():
                obs = obs_dict.get(aid, np.zeros(self.env_cfg.obs_dim))
                a, lp, v = agent.select_action(obs)
                actions[aid] = a
                log_probs[aid] = lp
                values[aid] = v

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            for aid, agent in self.agents.items():
                obs = obs_dict.get(aid, np.zeros(self.env_cfg.obs_dim))
                r = rewards.get(aid, 0.0)
                done = terms.get(aid, False) or truncs.get(aid, False)
                agent.store(obs, actions[aid], log_probs[aid], r, done, values[aid])
                episode_rewards[aid] += r

            obs_dict = next_obs
            if all(terms.get(a, False) or truncs.get(a, False) for a in self.agents):
                break

        return episode_rewards

    def update_all(self, obs_dict: dict) -> dict[str, dict]:
        metrics = {}
        for aid, agent in self.agents.items():
            last_obs = obs_dict.get(aid, np.zeros(self.env_cfg.obs_dim))
            metrics[aid] = agent.update(last_obs)
        return metrics

    def train_episode(self, env) -> tuple[dict, dict]:
        obs_dict, _ = env.reset()
        episode_rewards: dict[str, float] = {a: 0.0 for a in self.agents}
        step = 0

        while step < self.ppo_cfg.rollout_steps:
            actions, log_probs, values = {}, {}, {}
            for aid, agent in self.agents.items():
                obs = obs_dict.get(aid, np.zeros(self.env_cfg.obs_dim))
                a, lp, v = agent.select_action(obs)
                actions[aid] = a
                log_probs[aid] = lp
                values[aid] = v

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            for aid, agent in self.agents.items():
                obs = obs_dict.get(aid, np.zeros(self.env_cfg.obs_dim))
                r = rewards.get(aid, 0.0)
                done = terms.get(aid, False) or truncs.get(aid, False)
                agent.store(obs, actions[aid], log_probs[aid], r, done, values[aid])
                episode_rewards[aid] += r

            obs_dict = next_obs
            step += 1

            if all(terms.get(a, False) or truncs.get(a, False) for a in self.agents):
                obs_dict, _ = env.reset()

        metrics = self.update_all(obs_dict)
        return episode_rewards, metrics

    def save_all(self, checkpoint_dir: str, episode: int):
        os.makedirs(checkpoint_dir, exist_ok=True)
        for aid, agent in self.agents.items():
            path = os.path.join(checkpoint_dir, f"{aid.replace('_', '-')}_ep{episode}.pt")
            agent.save(path)
        # Also save a shared actor for the legacy loader
        first = next(iter(self.agents.values()))
        torch.save({"actor": first.actor.state_dict(), "critic": first.critic.state_dict()},
                   os.path.join(checkpoint_dir, "best_agent.pt"))

    def load_all(self, checkpoint_dir: str, episode: int):
        for aid, agent in self.agents.items():
            path = os.path.join(checkpoint_dir, f"{aid.replace('_', '-')}_ep{episode}.pt")
            if os.path.exists(path):
                agent.load(path)

    def load_best(self, path: str):
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location="cpu")
        for agent in self.agents.values():
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
        return True
