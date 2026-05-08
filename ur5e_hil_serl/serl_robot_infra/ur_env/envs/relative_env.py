"""Relative frame wrapper for UR5e.

This is adapted from hil-serl's RelativeFrame but uses a pure rotation matrix
(3×3) instead of the 6×6 transform matrix. This properly transforms forces,
torques, and velocities into the body frame.

Interface is identical to hil-serl's RelativeFrame — all downstream wrappers
(SERLObsWrapper, ChunkingWrapper, etc.) work unchanged.
"""

import copy
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env


def construct_homogeneous_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """Construct 4×4 homogeneous transform from [xyz, quat]."""
    T = np.eye(4)
    T[:3, 3] = pose_7d[:3]
    T[:3, :3] = R.from_quat(pose_7d[3:]).as_matrix()
    return T


def construct_rotation_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """Extract 3×3 rotation matrix from [xyz, quat]."""
    return R.from_quat(pose_7d[3:]).as_matrix()


class RelativeFrame(gym.Wrapper):
    """
    Transform observations and actions between base frame and end-effector frame.

    This wrapper:
    1. Transforms actions from body (EE) frame → base frame before env.step()
    2. Transforms observations from base frame → body (EE) frame after env.step()
    3. Optionally expresses tcp_pose relative to the reset pose

    Expected observation space:
    {
        "state": {
            "tcp_pose": Box(shape=(7,)),  # xyz + quat
            "tcp_vel": Box(shape=(6,)),
            "tcp_force": Box(shape=(3,)),
            "tcp_torque": Box(shape=(3,)),
            ...
        },
        ...
    }
    Action space: at least 6 DoF (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.rotation_matrix = np.eye(3)
        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            self.T_r_o_inv = np.eye(4)

    def step(self, action: np.ndarray):
        # Transform action from body frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        # Convert spacemouse intervention action back to body frame
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update rotation matrix from current pose
        self.rotation_matrix = construct_rotation_matrix(obs["state"]["tcp_pose"])

        # Transform observation to body frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        # Set rotation matrix from reset pose
        self.rotation_matrix = construct_rotation_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """Transform observations from base frame → body (EE) frame."""
        rot_inv = self.rotation_matrix.T

        # Velocities
        obs["state"]["tcp_vel"][:3] = rot_inv @ obs["state"]["tcp_vel"][:3]
        obs["state"]["tcp_vel"][3:] = rot_inv @ obs["state"]["tcp_vel"][3:]

        # Forces and torques
        obs["state"]["tcp_force"] = rot_inv @ obs["state"]["tcp_force"]
        obs["state"]["tcp_torque"] = rot_inv @ obs["state"]["tcp_torque"]

        # Relative pose
        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """Transform action from body (EE) frame → base frame."""
        action = np.array(action)  # in case of jax read-only array
        action[:3] = self.rotation_matrix @ action[:3]
        action[3:6] = self.rotation_matrix @ action[3:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """Transform action from base frame → body (EE) frame."""
        action = np.array(action)
        action[:3] = self.rotation_matrix.T @ action[:3]
        action[3:6] = self.rotation_matrix.T @ action[3:6]
        return action
