"""Bin relocation (pick-place) config for UR5e + Hand-E + RealSense + Kinect.

This mirrors hil-serl's usb_pickup_insertion but for our UR5e setup.
Gripper control is learned (single-arm-learned-gripper mode).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from ur_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperPenaltyWrapper,
)
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.ur5e_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.bin_relocation.wrapper import BinRelocationEnv


class EnvConfig(DefaultEnvConfig):
    """UR5e bin relocation hardware config."""

    # === Robot ===
    ROBOT_IP = "172.22.1.139"
    CONTROLLER_HZ = 100

    # === Cameras ===
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "034422070605",
            "dim": (640, 480),
            "exposure": 40000,
        },
    }
    KINECT_CAMERAS = {
        "overview": "000631452147",
    }
    IMAGE_CROP = {
        # Crop overview to focus on workspace
        # "overview": lambda img: img[200:700, 400:1200],
    }
    DISPLAY_IMAGE = True

    # === Reset pose (above pick location) ===
    # TODO: Measure actual pick-place reset position
    RESET_Q = np.deg2rad([15.0, -70.0, -120.0, -80.0, 90.0, 15.0])

    # === Target: TCP pose at place location ===
    # TODO: Measure actual target
    TARGET_POSE = np.array([0.40, 0.10, 0.10, 2.2, -2.24, 0.0])
    REWARD_THRESHOLD = np.array([0.03, 0.03, 0.03, 0.2, 0.2, 0.2])

    # === Safe home ===
    HOME_Q = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])

    # === Gripper: releases on reset (pick-place task) ===
    GRIPPER_RELEASE_ON_RESET = True
    GRIPPER_SLEEP = 0.5

    # === Safety box (larger for pick-place) ===
    ABS_POSE_LIMIT_LOW = np.array([0.05, -0.20, -0.01, -0.4, -0.4, -0.4])
    ABS_POSE_LIMIT_HIGH = np.array([0.50, 0.20, 0.25, 0.4, 0.4, 0.4])

    # === Action scaling (larger steps for pick-place) ===
    ACTION_SCALE = np.array([0.015, 0.08, 1.0])  # [pos_m, rot_rad, grip]

    # === Random reset ===
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05

    # === Impedance ===
    ERROR_DELTA = 0.04
    FORCEMODE_DAMPING = 0.05
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    # === Episode (longer for pick-place) ===
    MAX_EPISODE_LENGTH = 120


class TrainConfig(DefaultTrainingConfig):
    """Training config for bin relocation."""

    image_keys = ["wrist_1", "overview"]
    classifier_keys = ["overview"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    buffer_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = BinRelocationEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                logit = classifier_fn(obs)
                if hasattr(logit, "shape") and logit.shape != ():
                    logit = logit.squeeze()
                return int(sigmoid(logit).item() > 0.7)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
