"""Wrappers for UR5e HIL-SERL.

These mirror hil-serl's franka_env/envs/wrappers.py interface exactly,
so all hil-serl training scripts work unchanged.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R
from typing import List

from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from ur_env.spacemouse.fake_spacemouse import FakeSpaceMouseExpert

sigmoid = lambda x: 1 / (1 + np.exp(-x))


##############################################################################
# REWARD WRAPPERS
##############################################################################


class HumanClassifierWrapper(gym.Wrapper):
    """Ask human for binary reward at episode end."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info["succeed"] = rew
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """Use a learned classifier to compute binary reward from images.

    Interface identical to hil-serl's wrapper.
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        # Override reward with classifier (ignore base env's distance reward)
        classifier_rew = self.compute_reward(obs)
        # Only terminate from classifier reward or time limit/safety — NOT from
        # base env's distance-based reward (which causes false termination)
        base_env_reward_fired = info.get("succeed", False)
        if base_env_reward_fired and not classifier_rew:
            # Base env said "success" but classifier disagrees → don't terminate
            done = False
        done = done or bool(classifier_rew)
        rew = classifier_rew
        info["succeed"] = bool(classifier_rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    """Multi-stage reward: all classifiers must fire for success."""

    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)

    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue
            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1
        return sum(rewards)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or all(self.received)
        info["succeed"] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info["succeed"] = False
        return obs, info


##############################################################################
# OBSERVATION WRAPPERS
##############################################################################


class Quat2EulerWrapper(gym.ObservationWrapper):
    """Convert tcp_pose from [xyz, quat] (7,) to [xyz, euler_xyz] (6,).

    Interface identical to hil-serl's Quat2EulerWrapper.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


##############################################################################
# ACTION WRAPPERS
##############################################################################


class GripperCloseEnv(gym.ActionWrapper):
    """Force gripper closed — for peg/insertion tasks where object is pre-grasped.

    Reduces action space from 7D to 6D (removes gripper dimension).
    Interface identical to hil-serl's GripperCloseEnv.
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        new_action[6] = -1.0  # Close gripper (Hand-E: -1 = close)
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


##############################################################################
# SPACEMOUSE INTERVENTION
##############################################################################


class SpacemouseIntervention(gym.ActionWrapper):
    """Allow human intervention via SpaceMouse during training.

    Interface identical to hil-serl's SpacemouseIntervention.
    When the spacemouse is moved, its action replaces the policy action.
    Button presses control the gripper.
    """

    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices
        self.last_intervene = 0
        self.deadspace = 0.15

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:  policy action
        Output: spacemouse action if nonzero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False

        # Apply deadspace
        positive = np.clip(
            (expert_a - self.deadspace) / (1.0 - self.deadspace), 0.0, 1.0
        )
        negative = np.clip(
            (expert_a + self.deadspace) / (1.0 - self.deadspace), -1.0, 0.0
        )
        expert_a = positive + negative

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            self.last_intervene = time.time()
            return expert_a, True

        # Keep intervening for 0.5s after last input (smooth handover)
        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


class KeyboardIntervention(gym.ActionWrapper):
    """Allow human intervention via keyboard during training.

    Uses FakeSpaceMouseExpert from voxel-serl (proven working).
    Same interface as SpacemouseIntervention.

    Key mappings (arrow keys):
      Arrow Up/Down:    Forward/Backward (Y-axis)
      Arrow Left/Right: Left/Right (X-axis)
      1: Up (Z-axis)
      0: Down (Z-axis)
      Right Ctrl: Toggle gripper close/open
    """

    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = FakeSpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices
        self.last_intervene = 0

    def action(self, action: np.ndarray) -> np.ndarray:
        """Get keyboard action. If nonzero, overrides policy action."""
        expert_a, buttons = self.expert.get_action()
        self.left = bool(buttons[0])
        self.right = bool(buttons[1])
        intervened = False

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.array([-1.0])
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.array([1.0])
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            self.last_intervene = time.time()
            return expert_a, True

        # Keep intervening for 0.5s after last input
        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


##############################################################################
# GRIPPER PENALTY
##############################################################################


class GripperPenaltyWrapper(gym.RewardWrapper):
    """Penalize unnecessary gripper state changes.

    Interface identical to hil-serl's GripperPenaltyWrapper.
    """

    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0] if obs["state"].ndim > 1 else 0.0
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        return reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = (
            observation["state"][0, 0]
            if isinstance(observation["state"], np.ndarray)
            and observation["state"].ndim > 1
            else observation["state"].get("gripper_pose", np.array([0.0]))[0]
        )
        return observation, reward, terminated, truncated, info
