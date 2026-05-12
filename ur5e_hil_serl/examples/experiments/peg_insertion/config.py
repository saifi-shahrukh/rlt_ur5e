"""Peg insertion config for UR5e + Hand-E + RealSense + Kinect.

This mirrors hil-serl's ram_insertion/config.py but configured for our hardware.
The peg is pre-grasped, gripper stays closed throughout the episode.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

from ur_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    KeyboardIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)

# Default: use keyboard for teleoperation (no SpaceMouse needed)
# Change to SpacemouseIntervention when SpaceMouse is available
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.ur5e_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.peg_insertion.wrapper import PegInsertionEnv


class EnvConfig(DefaultEnvConfig):
    """UR5e peg insertion hardware config."""

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
        # Crop wrist camera to focus on peg/hole area
        # "wrist_1": lambda img: img[100:400, 150:500],
        # "overview": lambda img: img[200:600, 500:1000],
    }
    DISPLAY_IMAGE = True

    # === Reset pose (joint angles in radians) ===
    # Peg ~5cm above hole, gripper closed
    RESET_Q = np.deg2rad([34, -75.20, -130.69, -64.11, 90, 35]).reshape(1, -1)

    # === Target: TCP pose when peg is fully inserted ===
    TARGET_POSE = np.array([0.36066, 0.08130, 0.090, 2.200, -2.238, 0.006])  # position in m, rotation in MRP (≈ [126°, 128°, 0.3°])

    # === Reward threshold (not used when classifier is active) ===
    REWARD_THRESHOLD = np.array([0.010, 0.010, 0.010, 0.05, 0.05, 0.05])  # position in m, rotation in rad

    # === Safe home ===
    HOME_Q = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])

    # === Gripper: stays closed (peg is pre-grasped) ===
    GRIPPER_RELEASE_ON_RESET = False
    GRIPPER_SLEEP = 0.3

    # === Safety box ===
    # TIGHTENED to prevent forceMode singularity failures during RL exploration.
    # Position: keeps TCP >28cm from base (outside singularity envelope).
    # Orientation: ±0.10 MRP ≈ ±22° (prevents wrist singularity from J5 drift).
    # OLD (caused singularity): [0.10, -0.10, -0.01, -0.3, -0.3, -0.3]
    ABS_POSE_LIMIT_LOW = np.array([0.28, -0.02, 0.03, -0.10, -0.10, -0.10])
    ABS_POSE_LIMIT_HIGH = np.array([0.42, 0.14, 0.20, 0.10, 0.10, 0.10])

    # === Action scaling ===
    ACTION_SCALE = np.array([0.005, 0.03, 1.0], dtype=np.float32)  #[pos_m, rot_rad, grip] for robot movement.

    # === Random reset ===
    RANDOM_RESET = False                                                                                                                                                                  
    RANDOM_XY_RANGE = 0.015   # ±15mm                                                                                                                                                    
    RANDOM_RZ_RANGE = 0.03    # ±1.7°   

    # === Impedance parameters ===
    ERROR_DELTA = 0.03
    FORCEMODE_DAMPING = 0.1
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)

    # === Episode ===
    MAX_EPISODE_LENGTH = 300


class TrainConfig(DefaultTrainingConfig):
    """Training config for peg insertion."""

    image_keys = ["wrist_1", "overview"]
    classifier_keys = ["wrist_1", "overview"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    # Set to "keyboard" to use keyboard teleoperation, "spacemouse" for SpaceMouse
    intervention_mode: str = "keyboard"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = PegInsertionEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            if self.intervention_mode == "keyboard":
                env = KeyboardIntervention(env)
            else:
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

            # Require N consecutive positive frames to avoid false positives
            # (peg hovering above hole looks similar to peg inserted)
            consecutive_count = [0]  # mutable closure
            CONSECUTIVE_NEEDED = 3
            THRESHOLD = 0.70

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                logit = classifier_fn(obs)
                # classifier may return a 1D array — squeeze to scalar
                if hasattr(logit, 'shape') and logit.shape != ():
                    logit = logit.squeeze()
                prob = sigmoid(logit).item()
                if prob > THRESHOLD:
                    consecutive_count[0] += 1
                else:
                    consecutive_count[0] = 0
                if consecutive_count[0] >= CONSECUTIVE_NEEDED:
                    print(f"[CLASSIFIER] REWARD=1 after {CONSECUTIVE_NEEDED} consecutive frames (logit={float(logit):.3f}, prob={prob:.3f})")
                    consecutive_count[0] = 0  # reset after firing
                    return 1
                if prob > 0.5:
                    print(f"[CLASSIFIER] logit={float(logit):.3f}, prob={prob:.3f}, streak={consecutive_count[0]}/{CONSECUTIVE_NEEDED}")
                return 0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
