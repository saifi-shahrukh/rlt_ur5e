"""Configuration for the Kinect v2 camera plugin."""

from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("kinect")
@dataclass
class KinectCameraConfig(CameraConfig):
    """Kinect v2 (Xbox One) camera configuration.

    Uses pylibfreenect2 to capture frames from the Kinect v2.
    Native resolution: 1920x1080 (color), output resized to width x height.

    Attributes:
        serial: Kinect serial number (empty = first device)
        flip_horizontal: Flip image horizontally (Kinect mirrors by default)
    """
    serial: str = "000631452147"
    flip_horizontal: bool = True
    color_mode: ColorMode = ColorMode.RGB
