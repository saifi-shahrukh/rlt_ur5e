"""UR5E config with freedrive support and camera options.

Camera modes:
  - Single cam (default): RealSense D435 wrist only
  - Dual cam: RealSense D435 wrist + Kinect v2 Xbox overhead (via OpenCV)

Hardware:
  - RealSense D435 wrist camera serial: 034422070605
  - Kinect v2 Xbox overhead serial: 000631452147 (exposed as OpenCV device)
"""

from dataclasses import dataclass, field
from lerobot.cameras.configs import ColorMode
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot_camera_kinect import KinectCameraConfig


def _single_cam_cameras() -> dict[str, CameraConfig]:
    """Single camera: RealSense D435 wrist only."""
    return {
        "wrist_cam": RealSenseCameraConfig(
            serial_number_or_name="034422070605",
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
    }


def _dual_cam_cameras() -> dict[str, CameraConfig]:
    """Dual camera: RealSense D435 wrist + Kinect v2 overhead."""
    return {
        "wrist_cam": RealSenseCameraConfig(
            serial_number_or_name="034422070605",
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "overview_cam": KinectCameraConfig(
            serial="000631452147",
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            flip_horizontal=True,
        ),
    }


@RobotConfig.register_subclass("ur5e")
@dataclass
class UR5EConfig(RobotConfig):
    ip: str = "172.22.1.139"
    freedrive: bool = False
    calibrate_gripper: bool = False  # Skip gripper calibration by default

    cameras: dict[str, CameraConfig] = field(default_factory=_single_cam_cameras)


@RobotConfig.register_subclass("ur5e_dual_cam")
@dataclass
class UR5EDualCamConfig(RobotConfig):
    """UR5E with dual cameras (RealSense wrist + Kinect/OpenCV overhead)."""
    ip: str = "172.22.1.139"
    freedrive: bool = False
    calibrate_gripper: bool = False  # Skip gripper calibration by default (peg already grasped)

    cameras: dict[str, CameraConfig] = field(default_factory=_dual_cam_cameras)