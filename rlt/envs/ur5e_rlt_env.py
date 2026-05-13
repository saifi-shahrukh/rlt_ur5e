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

from rlt.utils.ur5e_kinematics import UR5eKinematics


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
        self.max_episode_chunks = 30  # 10 chunks × 10 steps = 100 steps at 10Hz = 10s

        # State
        self._last_raw_obs = None
        self._current_z_rl = np.zeros(self.token_dim, dtype=np.float32)
        self._current_proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        self._current_ref_chunk = np.zeros(
            (self.chunk_size, self.action_dim), dtype=np.float32
        )
        # Track joint state for VLA queries and Jacobian computation
        self._current_joints = np.zeros(6, dtype=np.float32)
        self._current_gripper = 0.9686  # Default closed

        # Kinematics: converts joint deltas → Cartesian deltas for SERL env
        self._kinematics = UR5eKinematics(mode="dh")

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

    _vla_debug_printed = False  # class-level flag for one-time debug

    def _get_vla_reference(self, raw_obs: dict) -> np.ndarray:
        """Get reference actions for the RL agent.

        IMPORTANT: The SERL env uses Cartesian delta control (6D: xyz + rotation)
        while the VLA outputs joint angle deltas. These are incompatible.

        Current behavior:
        - If VLA is connected: query it but output is near-zero (broken rank=4 model)
        - In both cases: return zeros → SAC must learn the full action from scratch
        - When a working VLA is available (π0 rank=16 from HPC), we'll add
          joint→Cartesian conversion here

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
            # After SERLObsWrapper, obs format is:
            #   {"state": flat_array, "wrist_1": (H,W,3), "overview": (H,W,3)}
            # OR before SERLObsWrapper:
            #   {"images": {"wrist_1": ..., "overview": ...}, "state": {...}}
            # OpenPI expects: exterior_image_1_left, wrist_image_left, wrist_image_right
            # We have 2 cameras → duplicate wrist for left+right (same as training data)
            images = {}

            # Debug: print obs keys once to diagnose format
            if not UR5eRLTEnv._vla_debug_printed:
                obs_keys = list(raw_obs.keys()) if isinstance(raw_obs, dict) else type(raw_obs)
                print(f"[UR5eRLTEnv DEBUG] raw_obs keys: {obs_keys}")
                for k, v in raw_obs.items():
                    if isinstance(v, np.ndarray):
                        print(f"  {k}: ndarray shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, dict):
                        print(f"  {k}: dict with keys={list(v.keys())}")
                    else:
                        print(f"  {k}: {type(v).__name__}")
                UR5eRLTEnv._vla_debug_printed = True

            # Case 1: After SERLObsWrapper — images at top level
            if "wrist_1" in raw_obs or "overview" in raw_obs:
                if "overview" in raw_obs:
                    img = raw_obs["overview"]
                    if isinstance(img, np.ndarray):
                        if img.ndim == 4:
                            img = img[0]
                        images["exterior_image_1_left"] = img

                wrist_key = "wrist_1" if "wrist_1" in raw_obs else "wrist_2"
                if wrist_key in raw_obs:
                    img = raw_obs[wrist_key]
                    if isinstance(img, np.ndarray):
                        if img.ndim == 4:
                            img = img[0]
                        images["wrist_image_left"] = img
                        images["wrist_image_right"] = img.copy()

            # Case 2: Before SERLObsWrapper — images nested under "images" key
            elif "images" in raw_obs:
                img_dict = raw_obs["images"]
                if "overview" in img_dict:
                    img = img_dict["overview"]
                    if img.ndim == 4:
                        img = img[0]
                    images["exterior_image_1_left"] = img

                wrist_key = "wrist_1" if "wrist_1" in img_dict else "wrist_2"
                if wrist_key in img_dict:
                    img = img_dict[wrist_key]
                    if img.ndim == 4:
                        img = img[0]
                    images["wrist_image_left"] = img
                    images["wrist_image_right"] = img.copy()

            # Extract joint state for VLA query
            # The VLA was trained on joint angles (from controller.get_state()["Q"])
            # The SERL obs only has tcp_pose (Cartesian), so we read joints directly
            joints = self._current_joints
            gripper = self._current_gripper

            # Try to get joint angles from the underlying SERL env's controller
            if self.serl_env is not None and not self.fake_env:
                try:
                    # Navigate through wrappers to get the base env with controller
                    base_env = self.serl_env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    if hasattr(base_env, 'controller'):
                        ctrl_state = base_env.controller.get_state()
                        joints = ctrl_state["Q"][:6].astype(np.float32)
                        gripper_raw = ctrl_state["gripper"][0]  # normalized 0-1
                        # Convert to training format: 247/255 = 0.9686 when closed
                        gripper = 0.9686274528503418  # hardcoded closed (training value)
                except Exception as e:
                    pass  # Fall back to cached joints

            # Fallback: check raw_obs state dict
            if np.all(joints == 0) and "state" in raw_obs:
                state = raw_obs["state"]
                if isinstance(state, np.ndarray) and len(state) >= 6:
                    joints = state[:6].astype(np.float32)
                elif isinstance(state, dict) and "joint_position" in state:
                    joints = np.array(state["joint_position"][:6], dtype=np.float32)

            # Call VLA server via websocket
            actions = self.vla_client.get_actions(
                joint_position=joints,
                gripper_position=gripper,
                images=images,
            )
            # actions: (action_horizon, 7) — ABSOLUTE joint targets from server
            # Convert to joint DELTAS: delta_q = target_q - current_q
            # Then convert joint deltas to Cartesian via Jacobian for SERL env

            if actions is None or len(actions) == 0:
                return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

            # Take first chunk_size actions
            chunk_actions = actions[:self.chunk_size]  # (chunk_size, 7)
            if len(chunk_actions) < self.chunk_size:
                # Pad with last action if not enough
                pad = np.tile(chunk_actions[-1:], (self.chunk_size - len(chunk_actions), 1))
                chunk_actions = np.concatenate([chunk_actions, pad], axis=0)

            # Compute joint deltas: VLA output is absolute targets
            # delta_q[i] = action_q[i] - current_q (for first step)
            # For subsequent steps: delta_q[i] = action_q[i] - action_q[i-1]
            joint_deltas = np.zeros((self.chunk_size, 6), dtype=np.float32)
            prev_q = joints.copy()
            for i in range(self.chunk_size):
                target_q = chunk_actions[i, :6]  # 6 joint angles
                joint_deltas[i] = target_q - prev_q
                prev_q = target_q

            # Convert joint deltas to SERL Cartesian actions via Jacobian
            ref_chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            q = joints.copy()
            for i in range(self.chunk_size):
                serl_action = self._kinematics.joint_delta_to_serl_action(joint_deltas[i], q)
                ref_chunk[i] = serl_action
                q = q + joint_deltas[i]  # update for next step

            # Debug: print first VLA reference once
            if not hasattr(self, '_vla_ref_printed'):
                dq_mag = np.linalg.norm(joint_deltas[0])
                ref_mag = np.linalg.norm(ref_chunk[0])
                print(f"[UR5eRLTEnv] VLA ref: joint_delta[0]={joint_deltas[0]} (|dq|={dq_mag:.4f} rad)")
                print(f"[UR5eRLTEnv] VLA ref: serl_action[0]={ref_chunk[0]} (|a|={ref_mag:.4f})")
                print(f"[UR5eRLTEnv] VLA raw action[0]={chunk_actions[0, :6]}")
                print(f"[UR5eRLTEnv] Current joints={joints}")
                self._vla_ref_printed = True

            return ref_chunk

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
        """Update cached joint state from controller (for VLA queries)."""
        # Best source: read joint angles directly from controller
        if self.serl_env is not None and not self.fake_env:
            try:
                base_env = self.serl_env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                if hasattr(base_env, 'controller'):
                    ctrl_state = base_env.controller.get_state()
                    self._current_joints = ctrl_state["Q"][:6].astype(np.float32)
                    return
            except Exception:
                pass

        # Fallback: try raw_obs
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
        # Reshape residual — in JOINT SPACE (matching VLA output)
        # residual is in [-1, 1], representing scaled joint angle deltas
        residual_chunk = residual_flat.reshape(self.chunk_size, self.action_dim)

        # Scale residual to actual joint angle deltas
        # VLA q99 range: max ~0.048 rad per step. Use max_residual as scale factor.
        # max_residual_pos here represents max joint delta in radians
        joint_delta_scale = self.config.max_residual_pos  # rad
        residual_joint = residual_chunk * joint_delta_scale  # (C, 6) in radians

        # Combine: VLA reference (joint delta) + SAC residual (joint delta)
        joint_chunk = self._current_ref_chunk + residual_joint  # still in joint space

        # Convert joint-space deltas → Cartesian-space deltas for SERL env
        # Uses Jacobian: dx = J(q) · dq
        q = self._current_joints.copy()
        final_chunk = np.zeros_like(joint_chunk)
        for i in range(self.chunk_size):
            serl_action = self._kinematics.joint_delta_to_serl_action(joint_chunk[i], q)
            final_chunk[i] = serl_action
            # Update q for next step in chunk (approximate: q += dq)
            q = q + joint_chunk[i]

        # Debug: log first action of first chunk in episode
        if self.episode_step == 0 and not UR5eRLTEnv._vla_debug_printed:
            print(f"[UR5eRLTEnv] Joint delta: {joint_chunk[0]} rad")
            print(f"[UR5eRLTEnv] SERL action: {final_chunk[0]} (Cartesian, [-1,1])")
            dx = self._kinematics.joint_to_cartesian(joint_chunk[0], self._current_joints)
            print(f"[UR5eRLTEnv] Cartesian motion: pos=[{dx[0]*1000:.2f},{dx[1]*1000:.2f},{dx[2]*1000:.2f}]mm")

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
        info["residual_norm"] = float(np.linalg.norm(joint_chunk))
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
