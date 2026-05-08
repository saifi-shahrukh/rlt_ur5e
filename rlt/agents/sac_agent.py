"""SAC Agent for RLT — State-based (no images).

This wraps the existing ur5e_hil_serl JAX SAC agent but configured for
state-based observations (z_rl + proprio + ref_chunk) instead of pixel inputs.

The key difference from standard SERL:
  - SERL SAC: takes images → ResNet encoder → MLP critic/actor
  - RLT SAC:  takes flat state [z_rl|proprio|ref] → MLP critic/actor

This is MUCH simpler and faster since we don't need an image encoder.

Paper reference:
  "We use SAC with RLPD-style symmetric sampling (50% demo, 50% online)
   and a BC regularizer β||a - ã||²"
"""
from __future__ import annotations

import os
import functools
from typing import Optional

import numpy as np

# Force JAX to CPU for actor (GPU reserved for VLA + RL Token PyTorch)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    pass  # Let user control this

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
    def __call__(self, obs, temperature=1.0):
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


class EnsembleCritic(nn.Module):
    """Ensemble of Q-functions (for pessimistic value estimation)."""
    hidden_dims: tuple = (256, 256)
    ensemble_size: int = 10

    @nn.compact
    def __call__(self, obs, action):
        # Vectorized ensemble using vmap
        VmapCritic = nn.vmap(
            MLPCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        return VmapCritic(hidden_dims=self.hidden_dims)(obs, action)


# ═══════════════════════════════════════════════════════════════════════════
# SAC Agent
# ═══════════════════════════════════════════════════════════════════════════


class RLTSACAgent:
    """SAC agent for RLT with BC regularization.

    Observation: [z_rl (512) | proprio (19) | ref_chunk_flat (60)] = 591
    Action: residual_chunk_flat (60) in [-1, 1]

    Loss:
        actor_loss = -Q(s,a) + β * ||a - 0||²  (regularize residual to be small)
        critic_loss = (Q(s,a) - target)²
        target = r + γ * (min_Q(s', a') - α * log_prob(a'))
    """

    def __init__(
        self,
        obs_dim: int = 591,
        action_dim: int = 60,
        hidden_dims: tuple = (256, 256),
        ensemble_size: int = 2,  # Paper: TD3-style, 2 Q-functions
        subsample_size: int = 2,  # Take min of both
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
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size

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

        # Critic ensemble
        self.critic_net = EnsembleCritic(
            hidden_dims=hidden_dims, ensemble_size=ensemble_size
        )
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
        self.target_entropy = -action_dim * 0.5  # heuristic

        self.rng = rng
        self._update_step = 0

    def sample_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action from the policy.

        Args:
            obs: (obs_dim,) observation vector
            deterministic: if True, return mean (no noise)

        Returns:
            action: (action_dim,) in [-1, 1]
        """
        self.rng, sample_rng = jax.random.split(self.rng)
        obs_jax = jnp.array(obs).reshape(1, -1)

        action = self._sample_action_jit(
            self.actor_state.params, obs_jax, sample_rng, deterministic
        )
        return np.array(action[0])

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def _sample_action_jit(self, actor_params, obs, rng, deterministic):
        mean, log_std = self.actor_net.apply(actor_params, obs)
        if deterministic:
            return jnp.tanh(mean)
        else:
            std = jnp.exp(log_std)
            noise = jax.random.normal(rng, shape=mean.shape)
            action = jnp.tanh(mean + std * noise)
            return action

    def update(self, batch: dict) -> dict:
        """Perform one SAC gradient update.

        Args:
            batch: dict with keys:
                - z_rl: (B, 512)
                - proprio: (B, 19)
                - action_chunk: (B, C, action_dim) — actual executed
                - ref_chunk: (B, C, action_dim) — VLA reference
                - reward: (B,)
                - z_rl_next: (B, 512)
                - proprio_next: (B, 19)
                - done: (B,)

        Returns:
            info: dict with loss values for logging
        """
        # Build flat observations
        B = batch["z_rl"].shape[0]
        ref_flat = batch["ref_chunk"].reshape(B, -1)
        obs = np.concatenate([batch["z_rl"], batch["proprio"], ref_flat], axis=-1)

        # For next obs, use ref_chunk as approximation (ref doesn't change much)
        obs_next = np.concatenate(
            [batch["z_rl_next"], batch["proprio_next"], ref_flat], axis=-1
        )

        # Action = what was actually executed as residual (normalized to [-1,1])
        # residual = action_chunk - ref_chunk, then normalize
        residual = batch["action_chunk"] - batch["ref_chunk"]  # (B, C, action_dim)
        action = residual.reshape(B, -1)  # (B, C*action_dim)
        # Clip to [-1, 1] (already should be since env clips)
        action = np.clip(action, -1, 1)

        reward = batch["reward"]
        done = batch["done"]

        # Convert to JAX
        obs_j = jnp.array(obs, dtype=jnp.float32)
        obs_next_j = jnp.array(obs_next, dtype=jnp.float32)
        action_j = jnp.array(action, dtype=jnp.float32)
        reward_j = jnp.array(reward, dtype=jnp.float32)
        done_j = jnp.array(done, dtype=jnp.float32)

        self.rng, update_rng = jax.random.split(self.rng)

        # Update
        info = self._update_jit(
            obs_j, action_j, reward_j, obs_next_j, done_j, update_rng
        )

        self._update_step += 1
        return {k: float(v) for k, v in info.items()}

    def _update_jit(self, obs, action, reward, obs_next, done, rng):
        """Non-jitted update (for simplicity — jit the inner loops if needed)."""
        alpha = jnp.exp(self.log_alpha)
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        # ── Critic update ─────────────────────────────────────────────────
        # Compute target Q
        next_mean, next_log_std = self.actor_net.apply(
            self.actor_state.params, obs_next
        )
        next_std = jnp.exp(next_log_std)
        noise = jax.random.normal(critic_rng, shape=next_mean.shape)
        next_action = jnp.tanh(next_mean + next_std * noise)

        # Log prob for entropy
        log_prob = self._log_prob(next_mean, next_log_std, next_action)

        # Target Q from ensemble (take min of random subsample)
        target_qs = self.critic_net.apply(
            self.target_critic_params, obs_next, next_action
        )  # (ensemble_size, B)

        # Subsample
        indices = jax.random.choice(
            rng, self.ensemble_size, shape=(self.subsample_size,), replace=False
        )
        target_qs_sub = target_qs[indices]  # (subsample_size, B)
        target_q = jnp.min(target_qs_sub, axis=0)  # (B,)

        target = reward + self.discount * (1 - done) * (
            target_q - alpha * log_prob
        )
        target = jax.lax.stop_gradient(target)

        # Critic loss
        def critic_loss_fn(critic_params):
            qs = self.critic_net.apply(critic_params, obs, action)  # (E, B)
            loss = jnp.mean((qs - target[None, :]) ** 2)
            return loss

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            self.critic_state.params
        )
        self.critic_state = self.critic_state.apply_gradients(grads=critic_grads)

        # ── Actor update ──────────────────────────────────────────────────
        def actor_loss_fn(actor_params):
            mean, log_std = self.actor_net.apply(actor_params, obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(actor_rng, shape=mean.shape)
            action_sample = jnp.tanh(mean + std * noise)
            log_prob = self._log_prob(mean, log_std, action_sample)

            # Q value of sampled action
            qs = self.critic_net.apply(self.critic_state.params, obs, action_sample)
            q_val = jnp.min(qs[:self.subsample_size], axis=0)

            # BC regularizer: penalize large residuals
            bc_loss = jnp.mean(action_sample ** 2)  # encourage residual → 0

            loss = jnp.mean(alpha * log_prob - q_val) + self.beta * bc_loss
            return loss, {"actor_loss": loss, "bc_loss": bc_loss, "q_val": jnp.mean(q_val)}

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor_state.params)
        self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)

        # ── Temperature update ────────────────────────────────────────────
        mean, log_std = self.actor_net.apply(self.actor_state.params, obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, shape=mean.shape)
        action_sample = jnp.tanh(mean + std * noise)
        log_prob = self._log_prob(mean, log_std, action_sample)

        alpha_loss = -jnp.mean(
            self.log_alpha * jax.lax.stop_gradient(log_prob + self.target_entropy)
        )
        alpha_grad = jax.grad(lambda la: -jnp.mean(
            la * jax.lax.stop_gradient(log_prob + self.target_entropy)
        ))(self.log_alpha)

        updates, self.alpha_opt_state = self.alpha_optimizer.update(
            alpha_grad, self.alpha_opt_state
        )
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)

        # ── Target network soft update ────────────────────────────────────
        self.target_critic_params = jax.tree.map(
            lambda p, tp: self.tau * p + (1 - self.tau) * tp,
            self.critic_state.params,
            self.target_critic_params,
        )

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_info["actor_loss"],
            "bc_loss": actor_info["bc_loss"],
            "q_val": actor_info["q_val"],
            "alpha": jnp.exp(self.log_alpha),
            "alpha_loss": alpha_loss,
        }

    @staticmethod
    def _log_prob(mean, log_std, action):
        """Compute log probability of tanh-squashed Gaussian."""
        std = jnp.exp(log_std)
        # Inverse tanh to get pre-squash
        pre_tanh = jnp.arctanh(jnp.clip(action, -0.999, 0.999))
        # Gaussian log prob
        log_prob = -0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
        # Tanh correction
        log_prob = log_prob - jnp.log(1 - action ** 2 + 1e-6)
        return jnp.sum(log_prob, axis=-1)

    def save(self, path: str):
        """Save agent state to file."""
        import pickle
        state = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
            "target_critic_params": self.target_critic_params,
            "log_alpha": self.log_alpha,
            "alpha_opt_state": self.alpha_opt_state,
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
        print(f"[SAC] Saved agent → {path}")

    def load(self, path: str):
        """Load agent state from file."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        # Rebuild train states with loaded params
        self.actor_state = self.actor_state.replace(params=state["actor_params"])
        self.critic_state = self.critic_state.replace(params=state["critic_params"])
        self.target_critic_params = state["target_critic_params"]
        self.log_alpha = state["log_alpha"]
        self.alpha_opt_state = state["alpha_opt_state"]
        self._update_step = state["update_step"]
        print(f"[SAC] Loaded agent ← {path} (step {self._update_step})")
