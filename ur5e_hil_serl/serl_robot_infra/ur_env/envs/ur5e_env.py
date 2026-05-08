"""Gym Interface for UR5e + Robotiq Hand-E + RealSense + Kinect v2.

This is the base environment that mirrors hil-serl's FrankaEnv interface.
It uses ur-rtde for direct robot control (no ROS dependency).

Key interface compatibility with FrankaEnv:
- observation_space: {"state": {...}, "images": {...}}
- action_space: Box(7,) → [dx, dy, dz, drx, dry, drz, gripper]
- step() returns (obs, reward, done, truncated, info)
- reset() returns (obs, info)
- Supports SpacemouseIntervention wrapper
- Supports RelativeFrame wrapper
- Supports Quat2EulerWrapper
"""

import time
import threading
import copy
import numpy as np
import gymnasium as gym
import cv2
import queue
from typing import Dict, Tuple
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture
from ur_env.utils.rotations import euler_2_quat, quat_2_euler

from robot_controllers.ur5e_controller import UrImpedanceController


class ImageDisplayer(threading.Thread):
    """Display camera images in a window (same as hil-serl)."""

    def __init__(self, queue, name="UR5e Cameras"):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            # Display at 320x320 for better visibility during teleoperation
            frame = np.concatenate(
                [cv2.resize(v, (320, 320)) for k, v in img_array.items() if "full" not in k],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for UR5eEnv.

    Interface-compatible with hil-serl's DefaultEnvConfig, but adapted for UR5e.
    Key differences:
    - No SERVER_URL (we use direct ur-rtde, not HTTP)
    - REALSENSE_CAMERAS uses serial strings (not dicts with dim/exposure)
    - Added KINECT_CAMERAS for overview camera
    - Added impedance parameters (ERROR_DELTA, FORCEMODE_*)
    - RESET_POSE is in joint space (RESET_Q) not Cartesian
    """

    # ===== Robot Connection =====
    ROBOT_IP: str = "172.22.1.139"
    CONTROLLER_HZ: int = 100

    # ===== Cameras =====
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": {
            "serial_number": "034422070605",
            "dim": (640, 480),
            "exposure": 40000,
        },
    }
    KINECT_CAMERAS: Dict = {
        "overview": "000631452147",
    }
    IMAGE_CROP: dict = {}  # Optional per-camera crop functions
    DISPLAY_IMAGE: bool = True

    # ===== Reset =====
    RESET_Q: np.ndarray = np.deg2rad([11.65, -75.15, -129.93, -64.88, 90.22, 12.72])
    HOME_Q: np.ndarray = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])
    RESET_POSE: np.ndarray = None  # If set, Cartesian reset instead of joint reset
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.0
    RANDOM_RZ_RANGE: float = 0.0

    # ===== Target & Reward =====
    TARGET_POSE: np.ndarray = np.zeros((6,))  # [x, y, z, rx, ry, rz]
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))

    # ===== Safety Box =====
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.array([0.5, 0.2, 0.3, 3.2, 0.2, 0.2])
    ABS_POSE_LIMIT_LOW: np.ndarray = np.array([0.0, -0.2, -0.01, 2.8, -0.2, -0.2])

    # ===== Action =====
    ACTION_SCALE: np.ndarray = np.array([0.01, 0.05, 1.0])  # [pos, rot, grip]

    # ===== Gripper =====
    GRIPPER_RELEASE_ON_RESET: bool = True
    GRIPPER_SLEEP: float = 0.3

    # ===== Impedance / Force Mode =====
    ERROR_DELTA: float = 0.03
    FORCEMODE_DAMPING: float = 0.1
    FORCEMODE_TASK_FRAME: np.ndarray = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR: np.ndarray = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS: np.ndarray = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    # ===== Episode =====
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0  # Joint reset every N episodes (0 = never)


##############################################################################


class UR5eEnv(gym.Env):
    """Gymnasium environment for UR5e + Hand-E.

    Designed to be a drop-in replacement for hil-serl's FrankaEnv.
    The observation/action interface is identical, so all hil-serl wrappers
    (RelativeFrame, Quat2Euler, SERLObsWrapper, ChunkingWrapper) work unchanged.
    """

    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
    ):
        if config is None:
            config = DefaultEnvConfig()

        self.config = config
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.hz = hz
        self.save_video = save_video
        self.cycle_count = 0
        self.curr_path_length = 0
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        # Convert RESET_Q to stored format
        self._reset_Q = np.array(config.RESET_Q, dtype=np.float64)
        if self._reset_Q.ndim == 1:
            self._reset_Q = self._reset_Q.reshape(1, -1)

        # Random reset params
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE

        # State variables
        self.currpos = np.zeros(7, dtype=np.float64)   # xyz + quat
        self.currvel = np.zeros(6, dtype=np.float64)
        self.currforce = np.zeros(3, dtype=np.float64)
        self.currtorque = np.zeros(3, dtype=np.float64)
        self.curr_gripper_pos = np.array([0.0])
        self.last_gripper_act = time.time()

        if save_video:
            print("Saving videos!")
            self.recording_frames = []

        # Boundary boxes for safety clipping
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        # MRP bounding box — clips RELATIVE orientation from reset pose
        # (same as ur5e_serl's mrp_bounding_box)
        self.mrp_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Reset pose — used for relative orientation clipping
        # Initialize with identity quaternion [x,y,z,w] = [0,0,0,1]
        self.curr_reset_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # Action space: [dx, dy, dz, drx, dry, drz, gripper] all in [-1, 1]
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        # Collect camera names for observation space
        self._rs_cameras = config.REALSENSE_CAMERAS
        self._kinect_cameras = getattr(config, "KINECT_CAMERAS", {})
        all_camera_keys = list(self._rs_cameras.keys()) + list(self._kinect_cameras.keys())

        # Observation space — matches hil-serl's FrankaEnv interface
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in all_camera_keys
                    }
                ),
            }
        )

        # Controller & cameras
        self.controller = None
        self.cap = None

        if fake_env:
            return

        # Initialize impedance controller
        self.controller = UrImpedanceController(
            robot_ip=config.ROBOT_IP,
            frequency=config.CONTROLLER_HZ,
            config=config,
            verbose=True,
        )
        self.controller.start()

        # Wait for controller to be ready
        while not self.controller.is_ready():
            time.sleep(0.1)
        print("[UR5eEnv] Controller ready!")

        # Initialize cameras
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        self.init_kinect_cameras(self._kinect_cameras)

        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, f"UR5e @ {config.ROBOT_IP}")
            self.displayer.start()

        # Update initial state
        self._update_currpos()

        # Keyboard listener for ESC termination (same as hil-serl)
        self.terminate = False
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except ImportError:
            pass

        print("[UR5eEnv] Initialized!")

    # ==================================================================
    # SAFETY CLIPPING
    # ==================================================================

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box.

        Uses MRP (Modified Rodrigues Parameters) to clip RELATIVE orientation
        from the reset pose — same logic as ur5e_serl's clip_safety_box().
        This avoids the euler angle singularity/wrapping issues that would
        otherwise destroy the robot's orientation.
        """
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        # Clip orientation as relative MRP from reset pose
        orientation_diff = (
            R.from_quat(pose[3:]) * R.from_quat(self.curr_reset_pose[3:]).inv()
        ).as_mrp()
        orientation_diff = np.clip(
            orientation_diff, self.mrp_bounding_box.low, self.mrp_bounding_box.high
        )
        pose[3:] = (
            R.from_mrp(orientation_diff) * R.from_quat(self.curr_reset_pose[3:])
        ).as_quat()
        return pose

    # ==================================================================
    # STEP
    # ==================================================================

    def step(self, action: np.ndarray) -> tuple:
        """Standard gym step. Interface identical to hil-serl's FrankaEnv."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute next target pose
        xyz_delta = action[:3]
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # Orientation: MRP delta (same as ur5e_serl)
        # c * r → applies rotation c after r
        self.nextpos[3:] = (
            R.from_mrp(action[3:6] * self.action_scale[1] / 4.0)
            * R.from_quat(self.currpos[3:])
        ).as_quat()

        # Gripper
        gripper_action = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_action)

        # Send clipped position command
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = (
            self.curr_path_length >= self.max_episode_length
            or reward
            or self.terminate
        )
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        """Default reward: check if TCP is within threshold of target.

        Uses self.currpos (absolute pose) — NOT the observation which may
        be transformed by wrappers (RelativeFrame, Quat2Euler).

        TARGET_POSE format: [x, y, z, rx, ry, rz] where rx/ry/rz is
        a rotation vector (axis-angle), as read from UR teach pendant.
        """
        # Use raw absolute pose (not transformed by wrappers)
        current_pos = self.currpos[:3]
        current_rot = R.from_quat(self.currpos[3:])

        target_pos = self._TARGET_POSE[:3]
        target_rot = R.from_rotvec(self._TARGET_POSE[3:])

        # Position error
        pos_err = np.abs(current_pos - target_pos)

        # Rotation error as rotation vector magnitude
        rot_err_mag = (current_rot.inv() * target_rot).magnitude()

        # Check thresholds: first 3 = position, last 3 = rotation
        pos_ok = np.all(pos_err < self._REWARD_THRESHOLD[:3])
        rot_ok = rot_err_mag < self._REWARD_THRESHOLD[3]  # use first rot threshold

        return pos_ok and rot_ok

    # ==================================================================
    # RESET
    # ==================================================================

    def go_to_reset(self, joint_reset=False):
        """Reset robot to start pose. Override in task subclasses for custom behavior."""
        # Joint reset if requested
        if joint_reset:
            print("[UR5eEnv] JOINT RESET")
            self.controller.move_to_joints(self.config.HOME_Q)
            time.sleep(0.5)

        # Select reset joints
        if self._reset_Q.shape[0] > 1:
            idx = self.cycle_count % self._reset_Q.shape[0]
            reset_q = self._reset_Q[idx]
        else:
            reset_q = self._reset_Q[0]

        # Move to reset joint configuration
        self.controller.move_to_joints(reset_q)
        time.sleep(0.3)

        # Update state after reset move
        self._update_currpos()

        # Store reset pose for relative orientation clipping
        self.curr_reset_pose[:] = self.currpos.copy()

        # Random perturbation on top of reset pose
        if self.randomreset:
            reset_pose = self.currpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler = R.from_quat(reset_pose[3:]).as_euler("xyz")
            euler[2] += np.random.uniform(-self.random_rz_range, self.random_rz_range)
            reset_pose[3:] = R.from_euler("xyz", euler).as_quat()
            self.interpolate_move(reset_pose, timeout=1.0)

    def reset(self, joint_reset=False, **kwargs):
        """Reset env. Interface identical to hil-serl's FrankaEnv.reset()."""
        self.last_gripper_act = time.time()

        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            joint_reset = True

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    # ==================================================================
    # ROBOT COMMANDS
    # ==================================================================

    def _send_pos_command(self, pos: np.ndarray):
        """Send target pose to impedance controller."""
        self.controller.set_target_pos(pos)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Send gripper command. Binary mode: <-0.5 close, >0.5 open."""
        if mode == "binary":
            now = time.time()
            if (
                pos <= -0.5
                and self.curr_gripper_pos[0] > 0.5
                and (now - self.last_gripper_act > self.gripper_sleep)
            ):
                self.controller.close_gripper()
                self.last_gripper_act = now
            elif (
                pos >= 0.5
                and self.curr_gripper_pos[0] < 0.5
                and (now - self.last_gripper_act > self.gripper_sleep)
            ):
                self.controller.open_gripper()
                self.last_gripper_act = now
        elif mode == "continuous":
            self.controller.set_gripper_pos(np.clip((pos + 1.0) / 2.0, 0, 1))

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Linear interpolation move (same interface as hil-serl)."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1.0 / self.hz)
        self._update_currpos()

    # ==================================================================
    # STATE UPDATE
    # ==================================================================

    def _update_currpos(self):
        """Get latest state from controller."""
        state = self.controller.get_state()
        self.currpos[:] = state["pos"]
        self.currvel[:] = state["vel"]
        self.currforce[:] = state["force"]
        self.currtorque[:] = state["torque"]
        self.curr_gripper_pos[:] = state["gripper"][0]  # normalized position

    # ==================================================================
    # OBSERVATIONS
    # ==================================================================

    def _get_obs(self) -> dict:
        """Get observation dict. Same structure as hil-serl's FrankaEnv._get_obs()."""
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos.copy(),
            "tcp_vel": self.currvel.copy(),
            "gripper_pose": self.curr_gripper_pos.copy(),
            "tcp_force": self.currforce.copy(),
            "tcp_torque": self.currtorque.copy(),
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    # ==================================================================
    # CAMERAS
    # ==================================================================

    def init_cameras(self, name_serial_dict=None):
        """Initialize RealSense cameras."""
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        if not name_serial_dict:
            return
        for cam_name, cam_cfg in name_serial_dict.items():
            if isinstance(cam_cfg, dict):
                serial = cam_cfg["serial_number"]
                dim = cam_cfg.get("dim", (640, 480))
                exposure = cam_cfg.get("exposure", 40000)
            else:
                serial = cam_cfg
                dim = (640, 480)
                exposure = 40000
            print(f"[Camera] Init RealSense '{cam_name}': {serial}")
            cap = VideoCapture(
                RSCapture(
                    name=cam_name,
                    serial_number=serial,
                    dim=dim,
                    fps=30,
                    exposure=exposure,
                )
            )
            self.cap[cam_name] = cap

    def init_kinect_cameras(self, name_serial_dict=None):
        """Initialize Kinect v2 cameras."""
        if not name_serial_dict:
            return
        if self.cap is None:
            self.cap = OrderedDict()
        try:
            from ur_env.camera.kinect_capture import KinectCapture

            for cam_name, cam_serial in name_serial_dict.items():
                print(f"[Camera] Init Kinect '{cam_name}': {cam_serial}")
                cap = VideoCapture(
                    KinectCapture(name=cam_name, serial_number=cam_serial)
                )
                self.cap[cam_name] = cap
        except ImportError as e:
            print(f"[Camera] Kinect skipped (pylibfreenect2 not installed): {e}")
        except Exception as e:
            print(f"[Camera] Kinect init failed: {e}")

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from all cameras. Same interface as hil-serl's FrankaEnv.get_im()."""
        images = {}
        display_images = {}
        full_res_images = {}

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                # Apply optional crop
                cropped_rgb = (
                    self.config.IMAGE_CROP[key](rgb)
                    if key in self.config.IMAGE_CROP
                    else rgb
                )
                resized = cv2.resize(
                    cropped_rgb,
                    self.observation_space["images"][key].shape[:2][::-1],
                )
                images[key] = resized[..., ::-1]  # BGR → RGB
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connection, press enter to relaunch..."
                )
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                self.init_kinect_cameras(self._kinect_cameras)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(display_images)

        return images

    def close_cameras(self):
        """Close all cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    # ==================================================================
    # VIDEO RECORDING
    # ==================================================================

    def save_video_recording(self):
        """Save recorded frames to video files."""
        import os

        try:
            if not hasattr(self, "recording_frames") or not self.recording_frames:
                return
            if not os.path.exists("./videos"):
                os.makedirs("./videos")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for camera_key in self.recording_frames[0].keys():
                video_path = f"./videos/{camera_key}_{timestamp}.mp4"
                first_frame = self.recording_frames[0][camera_key]
                height, width = first_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (width, height),
                )
                for frame_dict in self.recording_frames:
                    video_writer.write(frame_dict[camera_key])
                video_writer.release()
                print(f"Saved video: {video_path}")
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    # ==================================================================
    # CLEANUP
    # ==================================================================

    def close(self):
        """Clean shutdown."""
        if hasattr(self, "listener"):
            self.listener.stop()
        if self.controller:
            self.controller.stop()
        self.close_cameras()
        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join(timeout=2.0)
