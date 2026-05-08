"""Threaded video capture wrapper.

Provides a non-blocking read() interface for any camera backend
(RSCapture, KinectCapture). Runs a daemon thread that continuously
captures frames, so read() always returns the latest frame instantly.

Interface identical to hil-serl's VideoCapture.
"""

import threading
import queue
import time
import numpy as np


class VideoCapture:
    """Threaded wrapper around any camera with read()/close() interface."""

    def __init__(self, cap, max_queue_size=2):
        """
        Args:
            cap: Camera backend object with read() and close() methods.
                 read() should return (success, frame) or (success, frame, timestamp).
        """
        self.cap = cap
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        # Wait for first frame
        deadline = time.time() + 10.0
        while self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)

    def _reader(self):
        """Continuously read frames in background."""
        while self._running:
            try:
                result = self.cap.read()
                if isinstance(result, tuple) and len(result) >= 2:
                    success = result[0]
                    frame = result[1]
                else:
                    continue

                if success and frame is not None:
                    # Drop old frames to keep only latest
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put(frame)
            except Exception as e:
                if self._running:
                    time.sleep(0.01)
                else:
                    break

    def read(self):
        """Get the latest frame (blocking with timeout).

        Returns:
            np.ndarray: BGR image from camera.

        Raises:
            queue.Empty: if no frame available within 5 seconds.
        """
        return self._queue.get(timeout=5.0)

    def close(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if hasattr(self.cap, "close"):
            self.cap.close()
