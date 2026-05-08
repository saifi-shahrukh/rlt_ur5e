"""Keyboard-based teleoperation expert.

Uses pynput to read keyboard inputs for controlling the robot.
Provides the same get_action() interface as SpaceMouseExpert.

Key mappings:
  Translation: W/S (Y fwd/back), A/D (X left/right), Q/E (Z up/down)
  Rotation:    I/K (pitch), J/L (yaw), U/O (roll)
  Gripper:     Z (close), X (open)
  Speed:       1 (slow=0.3), 2 (medium=0.6), 3 (fast=1.0)
"""

import threading
import numpy as np

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class KeyboardExpert:
    """Keyboard teleoperation with same interface as SpaceMouseExpert."""

    def __init__(self):
        if not PYNPUT_AVAILABLE:
            raise ImportError("pynput not installed. Install with: pip install pynput")

        self._keys_pressed = set()
        self._lock = threading.Lock()
        self._speed = 0.6  # default medium speed
        self._left_button = False  # gripper close
        self._right_button = False  # gripper open

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

        print("[Keyboard Expert] Ready!")
        print("  Movement: W/S=Y, A/D=X, Q/E=Z")
        print("  Rotation: I/K=pitch, J/L=yaw, U/O=roll")
        print("  Gripper:  Z=close, X=open")
        print("  Speed:    1=slow, 2=medium, 3=fast")

    def _on_press(self, key):
        with self._lock:
            try:
                k = key.char.lower() if hasattr(key, 'char') and key.char else None
                if k:
                    self._keys_pressed.add(k)
                    # Speed control
                    if k == '1':
                        self._speed = 0.3
                    elif k == '2':
                        self._speed = 0.6
                    elif k == '3':
                        self._speed = 1.0
                    # Gripper buttons
                    elif k == 'z':
                        self._left_button = True
                    elif k == 'x':
                        self._right_button = True
            except AttributeError:
                pass

    def _on_release(self, key):
        with self._lock:
            try:
                k = key.char.lower() if hasattr(key, 'char') and key.char else None
                if k:
                    self._keys_pressed.discard(k)
                    if k == 'z':
                        self._left_button = False
                    elif k == 'x':
                        self._right_button = False
            except AttributeError:
                pass

    def get_action(self):
        """Get current keyboard state as action.

        Returns:
            action: np.ndarray of shape (6,) with values in [-1, 1]
                    [tx, ty, tz, rx, ry, rz]
            buttons: tuple of (left_pressed: bool, right_pressed: bool)
        """
        with self._lock:
            keys = self._keys_pressed.copy()
            speed = self._speed
            left = self._left_button
            right = self._right_button

        action = np.zeros(6, dtype=np.float32)

        # Translation
        if 'd' in keys:
            action[0] = speed   # +X
        if 'a' in keys:
            action[0] = -speed  # -X
        if 'w' in keys:
            action[1] = speed   # +Y
        if 's' in keys:
            action[1] = -speed  # -Y
        if 'e' in keys:
            action[2] = speed   # +Z (up)
        if 'q' in keys:
            action[2] = -speed  # -Z (down)

        # Rotation
        if 'u' in keys:
            action[3] = speed   # +roll
        if 'o' in keys:
            action[3] = -speed  # -roll
        if 'i' in keys:
            action[4] = speed   # +pitch
        if 'k' in keys:
            action[4] = -speed  # -pitch
        if 'j' in keys:
            action[5] = speed   # +yaw
        if 'l' in keys:
            action[5] = -speed  # -yaw

        buttons = (left, right)
        return action, buttons

    def stop(self):
        self._listener.stop()
