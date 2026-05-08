"""PCB insertion config for UR5e + Hand-E + RealSense + Kinect.

Similar to peg insertion but with tighter action scaling and safety box.
The PCB connector is pre-grasped, gripper stays closed.
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
from experiments.pcb_insertion.wrapper import PCBInsertionEnv


class EnvConfig(DefaultEnvConfig):
    """UR5e PCB insertion hardware config."""

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

    # === Reset pose (joint angles) ===
    # TODO: Measure actual PCB task reset position
    RESET_Q = np.deg2rad([11.65, -75.15, -129.93, -64.88, 90.22, 12.72])

    # === Target: TCP when PCB is fully inserted ===
    # TODO: Measure actual target
    TARGET_POSE = np.array([0.35, -0.05, 0.07, 2.2, -2.24, 0.0])
    REWARD_THRESHOLD = np.array([0.010, 0.010, 0.010, 0.08, 0.08, 0.08])

    # === Safe home ===
    HOME_Q = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])

    # === Gripper: stays closed ===
    GRIPPER_RELEASE_ON_RESET = False
    GRIPPER_SLEEP = 0.3

    # === Safety box (tighter than peg) ===
    ABS_POSE_LIMIT_LOW = np.array([0.12, -0.08, -0.01, -0.2, -0.2, -0.2])
    ABS_POSE_LIMIT_HIGH = np.array([0.38, 0.08, 0.15, 0.2, 0.2, 0.2])

    # === Action scaling (more precise than peg) ===
    ACTION_SCALE = np.array([0.005, 0.03, 1.0])  # [pos_m, rot_rad, grip]

    # === Random reset (smaller range for precision task) ===
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.02

    # === Impedance (softer for delicate parts) ===
    ERROR_DELTA = 0.02
    FORCEMODE_DAMPING = 0.15
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.3, 0.3, 0.3, 0.8, 0.8, 0.8])

    # === Episode ===
    MAX_EPISODE_LENGTH = 120


class TrainConfig(DefaultTrainingConfig):
    """Training config for PCB insertion."""

    image_keys = ["wrist_1", "overview"]
    classifier_keys = ["wrist_1", "overview"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = PCBInsertionEnv(
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
                return int(sigmoid(logit).item() > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
