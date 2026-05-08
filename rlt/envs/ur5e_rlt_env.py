"""
UR5e RLT Environment — Bridges VLA + RL Token + SERL Hardware.

This wraps the existing ur5e_hil_serl environment and adds:
  1. VLA inference (π0-FAST/π0/π0.5) for reference actions ã
  2. RL Token extraction (z_rl from RLTokenModel)
  3. Open-loop chunk execution (C=10 steps between RL decisions)
  4. Residual action application (final = VLA + residual)

The RL agent sees:
  obs = {"z_rl": (512,), "proprio": (19,), "ref_chunk": (C, action_dim)}

And outputs:
  residual_chunk: (C, action_dim)  — small corrections to VLA actions

The env executes:
  final_action[t] = ref_chunk[t] + clip(residual_chunk[t], max_residual)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np


class UR5eRLTEnv(gym.Env):
    """RLT environment wrapping the existing SERL UR5e setup.

    This environment:
    - Calls the VLA server at each chunk boundary to get reference actions
    - Extracts z_rl from VLM embeddings via the RL Token model
    - Executes C steps open-loop on the robot
    - Returns sparse reward from the reward classifier

    The RL agent only makes decisions every C=10 steps (1 second at 10Hz).
    Between decisions, actions execute open-loop.
    """

    def __init__(
        self,
        config,
        vla_hook=None,
        rl_token_model=None,
        serl_env=None,
        fake_env: bool = False,
    ):
        """
        Args:
            config: RLTConfig dataclass with all parameters
            vla_hook: Pi05Hook instance (None = no VLA, random reference)
            rl_token_model: RLTokenModel instance (None = zero z_rl)
            serl_env: Pre-built SERL environment (None = create from config)
            fake_env: If True, don't connect to hardware
        """
        super().__init__()
        self.config = config
        self.vla_hook = vla_hook
        self.rl_token_model = rl_token_model
        self.fake_env = fake_env

        # Dimensions
        self.token_dim = config.token_dim
        self.proprio_dim = config.proprio_dim
        self.action_dim = config.action_dim
        self.chunk_size = config.chunk_size

        # Build or use provided SERL environment
        if serl_env is not None:
            self.serl_env = serl_env
        else:
            self.serl_env = self._create_serl_env(fake_env)

        # Action space: residual corrections for one chunk
        # Each step's residual is in [-1, 1], scaled by max_residual
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.chunk_size * self.action_dim, dtype=np.float32),
            high=np.ones(self.chunk_size * self.action_dim, dtype=np.float32),
        )

        # Observation space: z_rl + proprio + flattened reference chunk
        obs_dim = self.token_dim + self.proprio_dim + (self.chunk_size * self.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode tracking
        self.episode_step = 0
        self.max_episode_chunks = config.total_episodes if hasattr(config, 'max_chunks_per_episode') else 10
        # Each episode: max 10 chunks × 10 steps = 100 steps (matches SERL)
        self.max_episode_chunks = 10

        # State
        self._last_raw_obs = None
        self._current_z_rl = np.zeros(self.token_dim, dtype=np.float32)
        self._current_proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        self._current_ref_chunk = np.zeros(
            (self.chunk_size, self.action_dim), dtype=np.float32
        )

    def _create_serl_env(self, fake_env: bool):
        """Create the standard SERL peg insertion environment.

        This reuses your existing working setup with all wrappers:
        GripperClose → KeyboardIntervention → RelativeFrame → Quat2Euler → SERLObs → Chunking
        """
        try:
            sys.path.insert(0, str(Path(__file__).parents[2] / "ur5e_hil_serl"))
            sys.path.insert(0, str(Path(__file__).parents[2] / "ur5e_hil_serl" / "serl_robot_infra"))
            sys.path.insert(0, str(Path(__file__).parents[2] / "ur5e_hil_serl" / "examples"))

            from experiments.mappings import CONFIG_MAPPING
            serl_config = CONFIG_MAPPING[self.config.task_name]()
            env = serl_config.get_environment(
                fake_env=fake_env,
                save_video=False,
                classifier=self.config.use_classifier,
            )
            return env
        except Exception as e:
            print(f"[UR5eRLTEnv] Warning: Could not create SERL env: {e}")
            print("[UR5eRLTEnv] Using dummy environment")
            return None

    def _get_vla_embeddings_and_reference(self, raw_obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Call VLA to get embeddings and reference actions.

        Args:
            raw_obs: SERL observation dict with images and state

        Returns:
            z_rl: (token_dim,) compressed RL token
            ref_chunk: (chunk_size, action_dim) VLA reference actions
        """
        if self.vla_hook is None:
            # No VLA — return zeros (for testing / warmup)
            z_rl = np.zeros(self.token_dim, dtype=np.float32)
            ref_chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            return z_rl, ref_chunk

        try:
            # Prepare observation for VLA
            vla_obs = {}
            if "images" in raw_obs:
                # SERL format: {"images": {"wrist_1": ..., "overview": ...}}
                for key, img in raw_obs["images"].items():
                    if img.ndim == 4:  # (1, H, W, C) stacked
                        img = img[0]
                    vla_obs[key] = img
            if "state" in raw_obs:
                state = raw_obs["state"]
                if isinstance(state, dict):
                    # Concat all state components
                    vla_obs["state"] = np.concatenate([
                        state.get("tcp_pose", np.zeros(6)),
                        state.get("tcp_vel", np.zeros(6)),
                        state.get("tcp_force", np.zeros(3)),
                        state.get("tcp_torque", np.zeros(3)),
                        state.get("gripper_pose", np.zeros(1)),
                    ])
                else:
                    vla_obs["state"] = state.flatten()

            # Get VLM embeddings + action chunk
            z_tokens, full_ref = self.vla_hook.get_embeddings_and_actions(
                vla_obs, prompt=self.config.language_instruction
            )
            # z_tokens: (N_prefix, 2048), full_ref: (H, 7)

            # Compress to RL token
            if self.rl_token_model is not None:
                import torch
                z_t = torch.tensor(z_tokens, dtype=torch.float32).unsqueeze(0)
                if next(self.rl_token_model.parameters()).is_cuda:
                    z_t = z_t.cuda()
                z_rl = self.rl_token_model.extract(z_t).squeeze(0).cpu().numpy()
            else:
                # No RL token model — use mean pooling as fallback
                z_rl = z_tokens.mean(axis=0)[:self.token_dim]

            # Extract reference chunk (first C actions, trim to action_dim)
            ref_chunk = full_ref[:self.chunk_size, :self.action_dim].copy()

            # Pad if VLA returned fewer actions than chunk_size
            if ref_chunk.shape[0] < self.chunk_size:
                pad = np.zeros((self.chunk_size - ref_chunk.shape[0], self.action_dim))
                ref_chunk = np.concatenate([ref_chunk, pad], axis=0)

            return z_rl.astype(np.float32), ref_chunk.astype(np.float32)

        except Exception as e:
            print(f"[UR5eRLTEnv] VLA inference failed: {e}")
            return (
                np.zeros(self.token_dim, dtype=np.float32),
                np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            )

    def _get_proprio(self, raw_obs: dict) -> np.ndarray:
        """Extract proprioceptive state from SERL observation."""
        if raw_obs is None:
            return np.zeros(self.proprio_dim, dtype=np.float32)

        if "state" in raw_obs:
            state = raw_obs["state"]
            if isinstance(state, np.ndarray):
                return state.flatten()[:self.proprio_dim].astype(np.float32)
            elif isinstance(state, dict):
                parts = []
                for key in ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]:
                    if key in state:
                        parts.append(np.array(state[key]).flatten())
                proprio = np.concatenate(parts) if parts else np.zeros(self.proprio_dim)
                return proprio[:self.proprio_dim].astype(np.float32)

        return np.zeros(self.proprio_dim, dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        """Build flat observation vector for RL agent."""
        return np.concatenate([
            self._current_z_rl,
            self._current_proprio,
            self._current_ref_chunk.flatten(),
        ]).astype(np.float32)

    def reset(self, **kwargs):
        """Reset environment and get initial VLA prediction."""
        self.episode_step = 0

        if self.serl_env is not None and not self.fake_env:
            raw_obs, info = self.serl_env.reset(**kwargs)
        else:
            # Fake env: generate dummy observation matching SERL format
            raw_obs = self._make_fake_obs()
            info = {}

        self._last_raw_obs = raw_obs

        # Get VLA prediction for initial state
        self._current_z_rl, self._current_ref_chunk = \
            self._get_vla_embeddings_and_reference(raw_obs)
        self._current_proprio = self._get_proprio(raw_obs)

        obs = self._build_obs()
        return obs, info

    def step(self, residual_flat: np.ndarray):
        """Execute one chunk (C steps) with residual corrections.

        Args:
            residual_flat: (C * action_dim,) flattened residual actions in [-1, 1]

        Returns:
            obs: next observation (after chunk execution)
            reward: total reward for the chunk
            terminated: whether episode ended (success or failure)
            truncated: whether max steps reached
            info: additional info
        """
        # Reshape residual
        residual_chunk = residual_flat.reshape(self.chunk_size, self.action_dim)

        # Scale residual by safety limits
        residual_scaled = np.zeros_like(residual_chunk)
        residual_scaled[:, :3] = residual_chunk[:, :3] * self.config.max_residual_pos
        residual_scaled[:, 3:6] = residual_chunk[:, 3:6] * self.config.max_residual_rot

        # Combine: final = VLA reference + residual
        final_chunk = self._current_ref_chunk + residual_scaled

        # Execute C steps open-loop
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        raw_obs = self._last_raw_obs

        for i in range(self.chunk_size):
            if self.serl_env is not None and not self.fake_env:
                raw_obs, reward, terminated, truncated, step_info = \
                    self.serl_env.step(final_chunk[i])
                total_reward += reward
                info.update(step_info)

                if terminated or truncated:
                    break
            else:
                # Fake env — simulate with random termination
                raw_obs = self._make_fake_obs()
                # Random success ~10% of episodes at last chunk
                if self.episode_step >= self.max_episode_chunks - 1 and i == self.chunk_size - 1:
                    if np.random.random() < 0.1:
                        total_reward = 1.0
                        terminated = True
                        break

        self._last_raw_obs = raw_obs
        self.episode_step += 1

        # Check max chunks
        if self.episode_step >= self.max_episode_chunks:
            truncated = True

        # Get new VLA prediction for next chunk
        if not (terminated or truncated):
            self._current_z_rl, self._current_ref_chunk = \
                self._get_vla_embeddings_and_reference(raw_obs)
        self._current_proprio = self._get_proprio(raw_obs)

        obs = self._build_obs()

        # Add RLT-specific info
        info["residual_norm"] = np.linalg.norm(residual_scaled)
        info["ref_norm"] = np.linalg.norm(self._current_ref_chunk)
        info["chunk_steps_executed"] = min(i + 1, self.chunk_size)

        return obs, total_reward, terminated, truncated, info

    def _make_fake_obs(self) -> dict:
        """Generate a fake observation for testing without hardware."""
        return {
            "state": np.random.randn(self.proprio_dim).astype(np.float32),
            "images": {
                "wrist_1": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                "overview": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            },
        }

    def get_raw_obs(self) -> dict:
        """Get the last raw SERL observation (for logging/visualization)."""
        return self._last_raw_obs

    def get_ref_chunk(self) -> np.ndarray:
        """Get current VLA reference chunk (for BC regularizer)."""
        return self._current_ref_chunk.copy()

    def close(self):
        """Clean up resources."""
        if self.serl_env is not None:
            self.serl_env.close()
        if self.vla_hook is not None:
            self.vla_hook.close()
