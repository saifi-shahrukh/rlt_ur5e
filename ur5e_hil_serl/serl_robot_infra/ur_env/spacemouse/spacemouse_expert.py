"""SpaceMouse expert interface for human teleoperation.

Provides the same get_action() interface as hil-serl's SpaceMouseExpert.
Returns (action_6d, buttons) where action_6d is [dx, dy, dz, drx, dry, drz]
normalized to [-1, 1] and buttons is (left_pressed, right_pressed).
"""

import numpy as np

try:
    import pyspacemouse
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SPACEMOUSE_AVAILABLE = False


class SpaceMouseExpert:
    """3Dconnexion SpaceMouse interface."""

    def __init__(self, device_number: int = 0):
        if not SPACEMOUSE_AVAILABLE:
            raise ImportError(
                "pyspacemouse not installed. Install with: pip install pyspacemouse hidapi"
            )

        success = pyspacemouse.open(
            dof_callback=None,
            button_callback=None,
            device_number=device_number,
        )
        if not success:
            raise RuntimeError(
                f"Failed to open SpaceMouse (device {device_number}). "
                "Check USB connection and permissions."
            )
        self.device_number = device_number

    def get_action(self):
        """Get current spacemouse state.

        Returns:
            action: np.ndarray of shape (6,) with values in [-1, 1]
                    [tx, ty, tz, rx, ry, rz] — translation and rotation
            buttons: tuple of (left_pressed: bool, right_pressed: bool)
        """
        state = pyspacemouse.read()

        # SpaceMouse axes: (x, y, z, roll, pitch, yaw)
        action = np.array([
            state.x,     # translate X (left/right)
            state.y,     # translate Y (forward/back)
            state.z,     # translate Z (up/down)
            state.roll,  # rotate X
            state.pitch, # rotate Y
            state.yaw,   # rotate Z
        ], dtype=np.float32)

        buttons = (bool(state.buttons[0]), bool(state.buttons[1]))

        return action, buttons


class FakeSpaceMouseExpert:
    """Dummy SpaceMouse that always returns zero actions.

    Used when no physical SpaceMouse is connected (e.g., on learner node).
    """

    def __init__(self, **kwargs):
        print("[SpaceMouse] Using FAKE SpaceMouse (no device connected)")

    def get_action(self):
        return np.zeros(6, dtype=np.float32), (False, False)
