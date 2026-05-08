"""Kinect v2 (Xbox One) camera implementation for LeRobot.

Uses pylibfreenect2 to capture color frames from the Kinect v2 sensor.
Native resolution: 1920x1080 BGRA, resized to configured width x height RGB.

Requires:
  - libfreenect2 installed (built at ~/freenect2)
  - pylibfreenect2 (pip install with LIBFREENECT2_INSTALL_PREFIX)
"""

import logging
import time
import warnings
from threading import Event, Lock, Thread
from typing import Any

# Suppress pkg_resources deprecation warning from pylibfreenect2
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_kinect import KinectCameraConfig

logger = logging.getLogger(__name__)


class KinectCamera(Camera):
    """Kinect v2 camera using pylibfreenect2."""

    def __init__(self, config: KinectCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial = config.serial
        self.flip_horizontal = config.flip_horizontal
        self.color_mode = config.color_mode

        self._fn = None
        self._device = None
        self._listener = None
        self._connected = False

        # Async read support
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._new_frame_event = Event()

    def __str__(self) -> str:
        return f"KinectCamera({self.serial})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect available Kinect v2 devices."""
        try:
            from pylibfreenect2 import Freenect2
            from pylibfreenect2 import createConsoleLogger, setGlobalLogger, LoggerLevel
            setGlobalLogger(createConsoleLogger(LoggerLevel.Error))
            fn = Freenect2()
            n = fn.enumerateDevices()
            devices = []
            for i in range(n):
                devices.append({
                    "index": i,
                    "serial": fn.getDeviceSerialNumber(i),
                    "type": "kinect_v2",
                })
            return devices
        except ImportError:
            logger.warning("pylibfreenect2 not installed")
            return []

    def connect(self, warmup: bool = True) -> None:
        """Open the Kinect v2 device and start color streaming."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType
        from pylibfreenect2 import createConsoleLogger, setGlobalLogger, LoggerLevel

        # Suppress verbose [Info] logs from libfreenect2
        setGlobalLogger(createConsoleLogger(LoggerLevel.Error))

        self._fn = Freenect2()
        num_devices = self._fn.enumerateDevices()

        if num_devices == 0:
            raise ConnectionError(
                "No Kinect v2 devices found. Check USB 3.0 connection and power."
            )

        # Find device by serial or use first
        target_serial = None
        for i in range(num_devices):
            s = self._fn.getDeviceSerialNumber(i)
            if isinstance(s, bytes):
                s = s.decode()
            if not self.serial or self.serial in s:
                target_serial = s
                break

        if target_serial is None:
            target_serial = self._fn.getDeviceSerialNumber(0)
            if isinstance(target_serial, bytes):
                target_serial = target_serial.decode()
            logger.warning(f"Kinect {self.serial} not found, using {target_serial}")

        # pylibfreenect2 expects bytes for openDevice
        if isinstance(target_serial, str):
            target_serial_bytes = target_serial.encode()
        else:
            target_serial_bytes = target_serial
        self._device = self._fn.openDevice(target_serial_bytes)
        self._listener = SyncMultiFrameListener(FrameType.Color)
        self._device.setColorFrameListener(self._listener)
        self._device.start()

        self._connected = True
        logger.info(f"KinectCamera({target_serial}) connected.")

        # Warmup - let auto-exposure stabilize
        if warmup:
            for _ in range(10):
                frames = self._listener.waitForNewFrame()
                self._listener.release(frames)
                time.sleep(0.03)

    def disconnect(self) -> None:
        """Stop and close the Kinect device."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        # Stop async thread if running
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            self._stop_event = None

        if self._device:
            self._device.stop()
            self._device.close()

        self._connected = False
        self._device = None
        self._listener = None
        logger.info(f"{self} disconnected.")

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Capture a single frame synchronously.

        Returns:
            np.ndarray: (height, width, 3) uint8 RGB or BGR image.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        frames = self._listener.waitForNewFrame()
        color = frames["color"]

        # Kinect color frame: BGRA (1080, 1920, 4)
        img = color.asarray()
        self._listener.release(frames)

        # Convert BGRA -> BGR
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Flip horizontally (Kinect mirrors by default)
        if self.flip_horizontal:
            bgr = cv2.flip(bgr, 1)

        # Resize to target resolution
        if self.width and self.height:
            if bgr.shape[0] != self.height or bgr.shape[1] != self.width:
                bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Convert to requested color mode
        requested = color_mode or self.color_mode
        if requested == ColorMode.RGB:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            return bgr

    def _read_loop(self) -> None:
        """Background read loop for async_read."""
        while not self._stop_event.is_set():
            try:
                frame = self.read()
                with self._frame_lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"KinectCamera read error: {e}")
                time.sleep(0.01)

    def async_read(self, timeout_ms: float = 400) -> NDArray[Any]:
        """Get the latest frame from background capture thread."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        # Start thread if not running
        if self._thread is None or not self._thread.is_alive():
            self._stop_event = Event()
            self._thread = Thread(target=self._read_loop, daemon=True, name="kinect_read")
            self._thread.start()

        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} timed out waiting for frame ({timeout_ms}ms).")

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} event set but no frame available.")

        return frame
