"""Cable routing config for UR5e + Hand-E + RealSense + Kinect.

Larger workspace and action scaling than insertion tasks.
Gripper stays closed (cable is pre-grasped).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from ur_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.ur5e_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.cable_routing.wrapper import CableRoutingEnv


class EnvConfig(DefaultEnvConfig):
    """UR5e cable routing hardware config."""

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
    IMAGE_CROP = {}
    DISPLAY_IMAGE = True

    # === Reset pose (cable start position) ===
    # TODO: Measure actual cable routing reset position
    RESET_Q = np.deg2rad([20.0, -65.0, -115.0, -90.0, 90.0, 20.0])

    # === Target: TCP when cable is fully routed ===
    # TODO: Measure actual target (last clip position)
    TARGET_POSE = np.array([0.45, 0.0, 0.10, 2.2, -2.2, 0.0])
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.15, 0.15, 0.15])

    # === Safe home ===
    HOME_Q = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])

    # === Gripper: stays closed (cable grasped) ===
    GRIPPER_RELEASE_ON_RESET = False
    GRIPPER_SLEEP = 0.3

    # === Safety box (larger for cable routing) ===
    ABS_POSE_LIMIT_LOW = np.array([0.05, -0.25, -0.01, -0.5, -0.5, -0.5])
    ABS_POSE_LIMIT_HIGH = np.array([0.55, 0.25, 0.25, 0.5, 0.5, 0.5])

    # === Action scaling (larger for cable routing sweeps) ===
    ACTION_SCALE = np.array([0.02, 0.1, 1.0])  # [pos_m, rot_rad, grip]

    # === Random reset ===
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.04

    # === Impedance (less damping for faster moves) ===
    ERROR_DELTA = 0.05
    FORCEMODE_DAMPING = 0.0
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    # === Episode ===
    MAX_EPISODE_LENGTH = 150


class TrainConfig(DefaultTrainingConfig):
    """Training config for cable routing."""

    image_keys = ["wrist_1", "overview"]
    classifier_keys = ["wrist_1", "overview"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    max_traj_length = 150

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = CableRoutingEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
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
                return int(sigmoid(logit).item() > 0.80)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
