"""Intel RealSense capture for UR5e HIL-SERL.

Supports the hil-serl camera config format:
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "034422070605",
            "dim": (640, 480),
            "exposure": 40000,
        },
    }
"""

import numpy as np
import pyrealsense2 as rs


class RSCapture:
    """RealSense camera capture with read()/close() interface."""

    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(
        self,
        name,
        serial_number,
        dim=(640, 480),
        fps=30,
        depth=False,
        exposure=40000,
    ):
        self.name = name
        available = self.get_device_serial_numbers()
        if serial_number not in available:
            raise RuntimeError(
                f"RealSense '{serial_number}' not found. Available: {available}"
            )
        self.serial_number = serial_number
        self.depth = depth

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)

        self.profile = self.pipe.start(self.cfg)

        # Set exposure
        sensor = self.profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.exposure, exposure)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

    def read(self):
        """Read a frame. Returns (success, image, timestamp).

        For hil-serl compatibility, the VideoCapture wrapper calls this
        and returns just the image.
        """
        frames = self.pipe.wait_for_frames(timeout_ms=5000)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame.is_video_frame():
            return False, None

        image = np.asarray(color_frame.get_data())

        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)

        return True, image

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
