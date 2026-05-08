"""Kinect v2 capture — threaded, segfault-safe, quiet."""
import os
import sys
import threading
import numpy as np
import cv2
import time
import atexit


def _suppress_freenect2_logs():
    """Suppress libfreenect2 C++ [Info] spam before any freenect2 import."""
    try:
        from pylibfreenect2 import createConsoleLogger, setGlobalLogger, LoggerLevel
        setGlobalLogger(createConsoleLogger(LoggerLevel.Error))
    except (ImportError, AttributeError):
        pass


class KinectCapture:
    """Kinect v2 with same read()/close() interface as RSCapture."""

    def __init__(self, name="kinect", serial_number=None, fps=30, rgb=True, depth=False):
        _suppress_freenect2_logs()
        from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType

        self.name = name
        self._fn2 = Freenect2()
        num_devices = self._fn2.enumerateDevices()
        if num_devices == 0:
            raise RuntimeError("No Kinect v2 found!")

        dev_index = 0
        if serial_number:
            serial_str = str(serial_number)
            found = False
            for i in range(num_devices):
                s = self._fn2.getDeviceSerialNumber(i)
                if isinstance(s, bytes):
                    s = s.decode("utf-8")
                if s == serial_str:
                    dev_index = i
                    found = True
                    break
            if not found:
                avail = []
                for i in range(num_devices):
                    s = self._fn2.getDeviceSerialNumber(i)
                    avail.append(s.decode("utf-8") if isinstance(s, bytes) else s)
                raise RuntimeError(f"Kinect '{serial_number}' not found. Available: {avail}")

        self._device = self._fn2.openDevice(dev_index)
        self._listener = SyncMultiFrameListener(FrameType.Color)
        self._device.setColorFrameListener(self._listener)
        self._device.start()

        self._lock = threading.Lock()
        self._latest_bgr = None
        self._latest_ts = 0.0
        self._running = True
        self._closed = False

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        deadline = time.time() + 5.0
        while self._latest_bgr is None and time.time() < deadline:
            time.sleep(0.05)

        if self._latest_bgr is None:
            self.close()
            raise RuntimeError("Kinect: no frames within 5s")

        atexit.register(self.close)
        print(f"[Kinect] Ready: {serial_number}")

    def _reader_loop(self):
        while self._running:
            try:
                frames = self._listener.waitForNewFrame()
                color = frames["color"]
                bgra = color.asarray().copy()
                self._listener.release(frames)

                bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                bgr = cv2.flip(bgr, 1)

                with self._lock:
                    self._latest_bgr = bgr
                    self._latest_ts = time.time()
            except Exception:
                if not self._running:
                    break
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._latest_bgr is not None:
                return True, self._latest_bgr.copy(), self._latest_ts
        return False, np.zeros((1080, 1920, 3), dtype=np.uint8), time.time()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._running = False
        time.sleep(0.2)
        try:
            self._device.stop()
        except Exception:
            pass
        try:
            self._device.close()
        except Exception:
            pass
