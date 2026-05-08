"""Keyboard-based SpaceMouse emulator.

Emulates a SpaceMouse using keyboard input. Uses the same get_action()
interface as the real SpaceMouseExpert so it drops in directly.

Key mappings:
    Arrow Up/Down:   Forward/Backward (Y-axis)
    Arrow Left/Right: Left/Right (X-axis)
    1: Up (Z-axis)
    0: Down (Z-axis)
    Right Ctrl: Toggle gripper (close/open)

From voxel-serl — proven working with our UR5e setup.
"""

import threading
import numpy as np
from typing import Tuple
from pynput import keyboard


class FakeSpaceMouseExpert:
    """
    Emulates a SpaceMouse using keyboard input.
    Arrow keys control movement:
        - Up/Down: Forward/Backward (Y-axis)
        - Left/Right: Left/Right (X-axis)
        - '1': Up (Z-axis)
        - '0': Down (Z-axis)
        - Right Ctrl: Toggle gripper close/open
    """

    def __init__(self):
        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 1]}

        # Start a thread to listen for keyboard input
        self.thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.thread.daemon = True
        self.thread.start()

        print("[FakeSpaceMouse] Keyboard control active!")
        print("  Arrow keys = XY movement")
        print("  1/0 = Z up/down")
        print("  Right Ctrl = toggle gripper")

    def _on_press(self, key):
        with self.state_lock:
            if key == keyboard.Key.up:
                self.latest_data["action"][0] = -1  # Forward (negative Y)
            elif key == keyboard.Key.down:
                self.latest_data["action"][0] = 1  # Backward (positive Y)
            elif key == keyboard.Key.left:
                self.latest_data["action"][1] = -1  # Left (negative X)
            elif key == keyboard.Key.right:
                self.latest_data["action"][1] = 1  # Right (positive X)
            elif hasattr(key, 'char') and key.char == '1':
                self.latest_data["action"][2] = 1  # Up (positive Z)
            elif hasattr(key, 'char') and key.char == '0':
                self.latest_data["action"][2] = -1  # Down (negative Z)
            elif key == keyboard.Key.ctrl_r:
                self.latest_data["buttons"] = [1, 0]  # close gripper

    def _on_release(self, key):
        with self.state_lock:
            if key in [keyboard.Key.up, keyboard.Key.down]:
                self.latest_data["action"][0] = 0
            elif key in [keyboard.Key.left, keyboard.Key.right]:
                self.latest_data["action"][1] = 0
            elif hasattr(key, 'char') and key.char in ['1', '0']:
                self.latest_data["action"][2] = 0
            elif key == keyboard.Key.ctrl_r:
                self.latest_data["buttons"] = [0, 1]  # open gripper

    def _listen_keyboard(self):
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            listener.join()

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state.

        Returns:
            action: np.ndarray of shape (6,) — [tx, ty, tz, rx, ry, rz]
            buttons: list of [left_button, right_button] (0 or 1)
        """
        with self.state_lock:
            return self.latest_data["action"].copy(), self.latest_data["buttons"][:]
