"""
UR5e RLT Environment — Bridges VLA + RL Token + SERL Hardware.

This wraps the existing ur5e_hil_serl environment and adds:
  1. VLA inference via WebSocket client (π0-FAST/π0/π0.5 server)
  2. RL Token extraction (z_rl from RLTokenModel) — optional
  3. Open-loop chunk execution (C=10 steps between RL decisions)
  4. Residual action application (final = VLA + residual)

The RL agent sees:
  obs = [z_rl (512) | proprio (19) | ref_chunk_flat (C*action_dim)]

And outputs:
  residual_flat: (C * action_dim,) in [-1, 1] — small corrections to VLA actions

The env executes:
  final_action[t] = ref_chunk[t] + clip(residual[t] * max_residual)

Architecture (separate processes):
  Terminal 1: VLA server (openpi venv, Python 3.11, JAX+GPU)
  Terminal 2: RLT training (ur5e_hil_serl venv, Python 3.10, JAX CPU + PyTorch GPU)
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
    - Calls the VLA WebSocket server at each chunk boundary for reference actions
    - Optionally extracts z_rl from VLM embeddings via the RL Token model
    - Executes C steps open-loop on the robot
    - Returns sparse reward from the reward classifier

    The RL agent only makes decisions every C=10 steps (1 second at 10Hz).
    Between decisions, actions execute open-loop.
    """

    def __init__(
        self,
        config,
        vla_client=None,
        rl_token_model=None,
        serl_env=None,
        fake_env: bool = False,
    ):
        """
        Args:
            config: RLTConfig dataclass with all parameters
            vla_client: VLAClient instance (websocket to VLA server)
            rl_token_model: RLTokenModel instance (None = zero z_rl)
            serl_env: Pre-built SERL environment (None = create from config)
            fake_env: If True, don't connect to hardware
        """
        super().__init__()
        self.config = config
        self.vla_client = vla_client
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
        self.max_episode_chunks = 10  # 10 chunks × 10 steps = 100 steps at 10Hz = 10s

        # State
        self._last_raw_obs = None
        self._current_z_rl = np.zeros(self.token_dim, dtype=np.float32)
        self._current_proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        self._current_ref_chunk = np.zeros(
            (self.chunk_size, self.action_dim), dtype=np.float32
        )
        # Track joint state for VLA queries
        self._current_joints = np.zeros(6, dtype=np.float32)
        self._current_gripper = 0.9686  # Default closed

    def _create_serl_env(self, fake_env: bool):
        """Create the standard SERL peg insertion environment.

        This reuses your existing working setup with all wrappers:
        GripperClose → RelativeFrame → Quat2Euler → SERLObs → Chunking
        """
        if fake_env:
            return None

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
            print(f"[UR5eRLTEnv] Falling back to dummy environment")
            return None

    def _get_vla_reference(self, raw_obs: dict) -> np.ndarray:
        """Call VLA server to get reference actions.

        Args:
            raw_obs: SERL observation dict with images and state

        Returns:
            ref_chunk: (chunk_size, action_dim) VLA reference actions (delta tcp)
        """
        if self.vla_client is None:
            # No VLA — return zeros (robot stays in place)
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        try:
            # Extract images from SERL observation
            images = {}
            if "images" in raw_obs:
                img_dict = raw_obs["images"]
                # Map SERL camera names to OpenPI names
                key_mapping = {
                    "wrist_1": "wrist_image_left",
                    "wrist_2": "wrist_image_right",
                    "overview": "exterior_image_1_left",
                }
                for serl_key, openpi_key in key_mapping.items():
                    if serl_key in img_dict:
                        img = img_dict[serl_key]
                        if img.ndim == 4:  # (1, H, W, C) stacked
                            img = img[0]
                        images[openpi_key] = img

            # Extract joint state
            joints = self._current_joints
            gripper = self._current_gripper

            if "state" in raw_obs:
                state = raw_obs["state"]
                if isinstance(state, np.ndarray) and len(state) >= 6:
                    # First 6 values are joint angles
                    joints = state[:6].astype(np.float32)
                elif isinstance(state, dict):
                    if "joint_position" in state:
                        joints = np.array(state["joint_position"][:6], dtype=np.float32)
                    elif "tcp_pose" in state:
                        # tcp_pose is (6,) [x,y,z,rx,ry,rz] - not joints!
                        # We need actual joint positions
                        pass

            # Call VLA server via websocket
            actions = self.vla_client.get_actions(
                joint_position=joints,
                gripper_position=gripper,
                images=images,
            )
            # actions: (action_horizon, 7) — absolute joint targets from server
            # The server already applies AbsoluteActions transform

            # Convert absolute targets to delta (relative to current joints)
            # The robot controller expects delta tcp, not absolute joints
            # But the VLA outputs absolute joint positions after AbsoluteActions
            # We need: delta = target - current
            ref_deltas = actions[:self.chunk_size, :self.action_dim] - joints[:self.action_dim]

            return ref_deltas.astype(np.float32)

        except Exception as e:
            print(f"[UR5eRLTEnv] VLA inference failed: {e}")
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

    def _get_proprio(self, raw_obs: dict) -> np.ndarray:
        """Extract proprioceptive state from SERL observation."""
        if raw_obs is None:
            return np.zeros(self.proprio_dim, dtype=np.float32)

        if "state" in raw_obs:
            state = raw_obs["state"]
            if isinstance(state, np.ndarray):
                proprio = state.flatten()[:self.proprio_dim]
                return proprio.astype(np.float32)
            elif isinstance(state, dict):
                parts = []
                for key in ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]:
                    if key in state:
                        parts.append(np.array(state[key]).flatten())
                if parts:
                    proprio = np.concatenate(parts)
                    return proprio[:self.proprio_dim].astype(np.float32)

        return np.zeros(self.proprio_dim, dtype=np.float32)

    def _update_joint_state(self, raw_obs: dict):
        """Update cached joint state from observation (for VLA queries)."""
        if raw_obs is None:
            return
        if "state" in raw_obs:
            state = raw_obs["state"]
            if isinstance(state, np.ndarray) and len(state) >= 6:
                self._current_joints = state[:6].astype(np.float32)
            elif isinstance(state, dict) and "joint_position" in state:
                self._current_joints = np.array(state["joint_position"][:6], dtype=np.float32)

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
            # Fake env: generate dummy observation
            raw_obs = self._make_fake_obs()
            info = {}

        self._last_raw_obs = raw_obs
        self._update_joint_state(raw_obs)

        # Get VLA reference actions for initial state
        self._current_ref_chunk = self._get_vla_reference(raw_obs)
        self._current_proprio = self._get_proprio(raw_obs)
        # z_rl stays zero without embeddings (RL Token needs VLM hook, not available via websocket)
        self._current_z_rl = np.zeros(self.token_dim, dtype=np.float32)

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
        self._update_joint_state(raw_obs)
        self.episode_step += 1

        # Check max chunks
        if self.episode_step >= self.max_episode_chunks:
            truncated = True

        # Get new VLA prediction for next chunk
        if not (terminated or truncated):
            self._current_ref_chunk = self._get_vla_reference(raw_obs)
        self._current_proprio = self._get_proprio(raw_obs)

        obs = self._build_obs()

        # Add RLT-specific info
        info["residual_norm"] = float(np.linalg.norm(residual_scaled))
        info["ref_norm"] = float(np.linalg.norm(self._current_ref_chunk))
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
            try:
                self.serl_env.close()
            except:
                pass
        if self.vla_client is not None:
            try:
                self.vla_client.close()
            except:
                pass
