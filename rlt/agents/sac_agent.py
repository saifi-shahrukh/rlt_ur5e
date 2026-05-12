"""SAC Agent for RLT — Properly JIT-compiled, functional style.

State-based SAC with BC regularization for residual learning.
All heavy computation is JIT-compiled via standalone functions.

The key difference from standard SERL:
  - SERL SAC: takes images → ResNet encoder → MLP critic/actor
  - RLT SAC:  takes flat state [z_rl|proprio|ref] → MLP critic/actor

Paper reference:
  "We use SAC with RLPD-style symmetric sampling (50% demo, 50% online)
   and a BC regularizer β||a - ã||²"
"""
from __future__ import annotations

import os
import pickle
import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


# ═══════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════


class MLPActor(nn.Module):
    """Gaussian actor MLP for continuous actions."""
    action_dim: int
    hidden_dims: tuple = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs):
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class MLPCritic(nn.Module):
    """Q-value critic MLP."""
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


class DoubleCritic(nn.Module):
    """Two Q-functions (TD3-style min)."""
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        q1 = MLPCritic(self.hidden_dims, name="q1")(obs, action)
        q2 = MLPCritic(self.hidden_dims, name="q2")(obs, action)
        return q1, q2


# ═══════════════════════════════════════════════════════════════════════════
# Functional helpers (JIT is applied inside the agent via closures)
# ═══════════════════════════════════════════════════════════════════════════


def _log_prob(mean, log_std, action):
    """Log probability of tanh-squashed Gaussian."""
    std = jnp.exp(log_std)
    pre_tanh = jnp.arctanh(jnp.clip(action, -0.999, 0.999))
    log_p = -0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
    log_p = log_p - jnp.log(1 - action ** 2 + 1e-6)
    return jnp.sum(log_p, axis=-1)


@jax.jit
def _soft_update(params, target_params, tau):
    """Soft update of target network."""
    return jax.tree.map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)


# ═══════════════════════════════════════════════════════════════════════════
# SAC Agent (stateful wrapper around JIT-compiled functions)
# ═══════════════════════════════════════════════════════════════════════════


class RLTSACAgent:
    """SAC agent for RLT with BC regularization.

    Observation: [z_rl (512) | proprio (19) | ref_chunk_flat (60)] = 591
    Action: residual_chunk_flat (60) in [-1, 1]
    """

    def __init__(
        self,
        obs_dim: int = 591,
        action_dim: int = 60,
        hidden_dims: tuple = (256, 256),
        ensemble_size: int = 2,
        subsample_size: int = 2,
        discount: float = 0.97,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        beta: float = 1.0,
        init_temperature: float = 0.1,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.beta = beta

        rng = jax.random.PRNGKey(seed)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Initialize networks
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        # Actor
        self.actor_net = MLPActor(action_dim=action_dim, hidden_dims=hidden_dims)
        actor_params = self.actor_net.init(actor_rng, dummy_obs)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor_net.apply,
            params=actor_params,
            tx=optax.adam(actor_lr),
        )

        # Critic (Double Q)
        self.critic_net = DoubleCritic(hidden_dims=hidden_dims)
        critic_params = self.critic_net.init(critic_rng, dummy_obs, dummy_action)
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic_net.apply,
            params=critic_params,
            tx=optax.adam(critic_lr),
        )
        self.target_critic_params = critic_params

        # Temperature (log_alpha)
        self.log_alpha = jnp.log(init_temperature)
        self.alpha_opt_state = optax.adam(temp_lr).init(self.log_alpha)
        self.alpha_optimizer = optax.adam(temp_lr)
        self.target_entropy = -action_dim * 0.5

        self.rng = rng
        self._update_step = 0
        self._jit_warmed = False

        # Build JIT-compiled closures that capture network apply fns
        self._build_jit_fns()

    def _build_jit_fns(self):
        """Create JIT-compiled functions as closures over network apply fns."""
        actor_apply = self.actor_net.apply
        critic_apply = self.critic_net.apply
        discount = self.discount
        beta = self.beta

        @jax.jit
        def sample_action_jit(actor_params, obs, rng):
            mean, log_std = actor_apply(actor_params, obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(rng, shape=mean.shape)
            return jnp.tanh(mean + std * noise)

        @jax.jit
        def sample_action_det_jit(actor_params, obs):
            mean, _ = actor_apply(actor_params, obs)
            return jnp.tanh(mean)

        @jax.jit
        def update_critic_jit(critic_params, target_critic_params, actor_params,
                              obs, action, reward, obs_next, done, rng, alpha):
            next_mean, next_log_std = actor_apply(actor_params, obs_next)
            next_std = jnp.exp(next_log_std)
            noise = jax.random.normal(rng, shape=next_mean.shape)
            next_action = jnp.tanh(next_mean + next_std * noise)
            next_log_prob = _log_prob(next_mean, next_log_std, next_action)

            tq1, tq2 = critic_apply(target_critic_params, obs_next, next_action)
            target_q = jnp.minimum(tq1, tq2)
            target = reward + discount * (1 - done) * (target_q - alpha * next_log_prob)
            target = jax.lax.stop_gradient(target)

            def loss_fn(params):
                q1, q2 = critic_apply(params, obs, action)
                loss = jnp.mean((q1 - target) ** 2) + jnp.mean((q2 - target) ** 2)
                return loss, jnp.mean(q1)

            (loss, q1_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_params)
            return loss, grads, q1_mean

        @jax.jit
        def update_actor_jit(actor_params, critic_params, obs, rng, alpha):
            def loss_fn(params):
                mean, log_std = actor_apply(params, obs)
                std = jnp.exp(log_std)
                noise = jax.random.normal(rng, shape=mean.shape)
                action = jnp.tanh(mean + std * noise)
                log_prob = _log_prob(mean, log_std, action)

                q1, q2 = critic_apply(critic_params, obs, action)
                q_min = jnp.minimum(q1, q2)

                bc_loss = jnp.mean(action ** 2)
                loss = jnp.mean(alpha * log_prob - q_min) + beta * bc_loss
                return loss, {"log_prob": jnp.mean(log_prob), "q_val": jnp.mean(q_min), "bc_loss": bc_loss}

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)
            return loss, grads, info

        @jax.jit
        def update_temp_jit(log_alpha, log_prob, target_entropy):
            alpha_loss = -jnp.mean(log_alpha * jax.lax.stop_gradient(log_prob + target_entropy))
            alpha_grad = jax.grad(lambda la: -jnp.mean(
                la * jax.lax.stop_gradient(log_prob + target_entropy)
            ))(log_alpha)
            return alpha_loss, alpha_grad

        self._sample_action_jit = sample_action_jit
        self._sample_action_det_jit = sample_action_det_jit
        self._update_critic_jit = update_critic_jit
        self._update_actor_jit = update_actor_jit
        self._update_temp_jit = update_temp_jit

    def warmup_jit(self):
        """Pre-compile JIT functions with dummy data. Call once at start."""
        if self._jit_warmed:
            return
        print("[SAC] JIT warmup (compiling)...")
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_action = jnp.zeros((1, self.action_dim))
        rng = jax.random.PRNGKey(0)

        _ = self._sample_action_jit(self.actor_state.params, dummy_obs, rng)
        _ = self._sample_action_det_jit(self.actor_state.params, dummy_obs)
        _ = self._update_critic_jit(
            self.critic_state.params, self.target_critic_params, self.actor_state.params,
            dummy_obs, dummy_action, jnp.zeros(1), dummy_obs, jnp.zeros(1),
            rng, jnp.exp(self.log_alpha),
        )
        _ = self._update_actor_jit(
            self.actor_state.params, self.critic_state.params,
            dummy_obs, rng, jnp.exp(self.log_alpha),
        )

        self._jit_warmed = True
        print("[SAC] JIT warmup complete.")

    def sample_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action from the policy."""
        obs_jax = jnp.array(obs, dtype=jnp.float32).reshape(1, -1)

        if deterministic:
            action = self._sample_action_det_jit(self.actor_state.params, obs_jax)
        else:
            self.rng, sample_rng = jax.random.split(self.rng)
            action = self._sample_action_jit(self.actor_state.params, obs_jax, sample_rng)
        return np.array(action[0])

    def update(self, batch: dict) -> dict:
        """Perform one SAC gradient update."""
        # Build flat observations
        B = batch["z_rl"].shape[0]
        ref_flat = batch["ref_chunk"].reshape(B, -1)
        obs = np.concatenate([batch["z_rl"], batch["proprio"], ref_flat], axis=-1)
        obs_next = np.concatenate(
            [batch["z_rl_next"], batch["proprio_next"], ref_flat], axis=-1
        )

        # Residual = action - ref, normalized to [-1, 1]
        residual = batch["action_chunk"] - batch["ref_chunk"]
        action = np.clip(residual.reshape(B, -1), -1, 1)

        reward = batch["reward"]
        done = batch["done"]

        # Convert to JAX
        obs_j = jnp.array(obs, dtype=jnp.float32)
        obs_next_j = jnp.array(obs_next, dtype=jnp.float32)
        action_j = jnp.array(action, dtype=jnp.float32)
        reward_j = jnp.array(reward, dtype=jnp.float32)
        done_j = jnp.array(done, dtype=jnp.float32)

        self.rng, critic_rng, actor_rng = jax.random.split(self.rng, 3)
        alpha = jnp.exp(self.log_alpha)

        # ── Critic update ─────────────────────────────────────────────
        critic_loss, critic_grads, q_mean = self._update_critic_jit(
            self.critic_state.params, self.target_critic_params, self.actor_state.params,
            obs_j, action_j, reward_j, obs_next_j, done_j,
            critic_rng, alpha,
        )
        self.critic_state = self.critic_state.apply_gradients(grads=critic_grads)

        # ── Actor update ──────────────────────────────────────────────
        actor_loss, actor_grads, actor_info = self._update_actor_jit(
            self.actor_state.params, self.critic_state.params,
            obs_j, actor_rng, alpha,
        )
        self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)

        # ── Temperature update ────────────────────────────────────────
        alpha_loss, alpha_grad = self._update_temp_jit(
            self.log_alpha, actor_info["log_prob"], self.target_entropy
        )
        updates, self.alpha_opt_state = self.alpha_optimizer.update(
            alpha_grad, self.alpha_opt_state
        )
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)

        # ── Target network soft update ────────────────────────────────
        self.target_critic_params = _soft_update(
            self.critic_state.params, self.target_critic_params, self.tau
        )

        self._update_step += 1

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "bc_loss": float(actor_info["bc_loss"]),
            "q_val": float(q_mean),
            "alpha": float(jnp.exp(self.log_alpha)),
            "alpha_loss": float(alpha_loss),
        }

    def save(self, path: str):
        """Save agent state."""
        state = {
            "actor_params": jax.device_get(self.actor_state.params),
            "critic_params": jax.device_get(self.critic_state.params),
            "target_critic_params": jax.device_get(self.target_critic_params),
            "log_alpha": float(self.log_alpha),
            "alpha_opt_state": jax.device_get(self.alpha_opt_state),
            "update_step": self._update_step,
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "discount": self.discount,
                "tau": self.tau,
                "beta": self.beta,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[SAC] Saved → {path} (step {self._update_step})")

    def load(self, path: str):
        """Load agent state."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.actor_state = self.actor_state.replace(params=state["actor_params"])
        self.critic_state = self.critic_state.replace(params=state["critic_params"])
        self.target_critic_params = state["target_critic_params"]
        self.log_alpha = jnp.array(state["log_alpha"])
        self.alpha_opt_state = state["alpha_opt_state"]
        self._update_step = state["update_step"]
        print(f"[SAC] Loaded ← {path} (step {self._update_step})")
