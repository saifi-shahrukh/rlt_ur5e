"""Keyboard Cartesian teleoperator for UR5e.

Two modes:
  cartesian  — WASD/RF keys drive the TCP; Jacobian IK returns joint targets
  freedrive  — human physically guides the robot; read-back joints as action

Key Mapping (cartesian mode — all from operator's perspective, standing in front of robot):
  Translation:
    W/S → −X/+X  (W=away from operator, S=toward operator)
    A/D → −Y/+Y  (A=operator's left, D=operator's right)
    Q/E → +Z/−Z  (Q=up, E=down)
  Rotation (clockwise = CW when looking at the robot along that axis):
    I/K → CW/ACW around X
    J/L → CW/ACW around Y
    U/O → CW/ACW around Z
  G → toggle gripper   +/- → adjust speed

Episode management (handled by record.py):
  SPACE → start recording   →  end episode   ←  rerecord   Esc  stop recording
"""

import logging
import time
import numpy as np
from threading import Lock

import rtde_receive as rtde_rec_mod

from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .config_keyboard_ur5e import KeyboardUR5eConfig
from .ur5e_kin import cartesian_to_joint_delta

logger = logging.getLogger(__name__)


class KeyboardUR5e(Teleoperator):
    config_class = KeyboardUR5eConfig
    name = "keyboard_ur5e"

    # key → (cartesian_axis_index, sign)
    # All directions are from OPERATOR's perspective (standing in front of robot, facing it).
    # Translations: W=away, S=toward, A=left, D=right, Q=up, E=down
    # Rotations: "clockwise" = CW when looking at the robot along that axis.
    #   Since robot +X points toward operator, +Rx = CCW from operator's view.
    #   So CW from operator = -Rx, -Ry, -Rz (all flipped).
    _KEY_MAP: dict[str, tuple[int, int]] = {
        "w": (0, -1), "s": (0, +1),   # X  W=away(-X) / S=toward(+X)
        "a": (1, -1), "d": (1, +1),   # Y  A=left(-Y) / D=right(+Y)
        "q": (2, +1), "e": (2, -1),   # Z  Q=up(+Z) / E=down(-Z)
        "i": (3, -1), "k": (3, +1),   # Rx I=CW(-Rx) / K=ACW(+Rx) around X
        "j": (4, -1), "l": (4, +1),   # Ry J=CW(-Ry) / L=ACW(+Ry) around Y
        "u": (5, -1), "o": (5, +1),   # Rz U=CW(-Rz) / O=ACW(+Rz) around Z
    }

    def __init__(self, config: KeyboardUR5eConfig):
        super().__init__(config)
        self.config = config
        self.rtde_rec = None          # separate RTDE receive (multiple allowed)

        self._lock = Lock()
        self._pressed: set[str] = set()
        self._gripper: float = 0.0    # 0 = open, 1 = closed
        self._speed_mult: float = 1.0
        self._listener = None
        self._last_time: float = 0.0
        self._last_log: float = 0.0

    # ── LeRobot Teleoperator interface ─────────────────────

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "joint_0": float, "joint_1": float, "joint_2": float,
            "joint_3": float, "joint_4": float, "joint_5": float,
            "gripper": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.rtde_rec is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Separate RTDE *receive* connection (UR allows multiple)
        self.rtde_rec = rtde_rec_mod.RTDEReceiveInterface(self.config.robot_ip)
        self._last_time = time.perf_counter()
        self._start_keyboard()

        mode = self.config.mode
        logger.info(f"{self} connected  mode={mode}")
        if mode == "cartesian":
            logger.info(
                "  Translation: W/S=±X(away/toward)  A/D=±Y(left/right)  Q/E=±Z(up/down)\n"
                "  Rotation:    I/K=CW/ACW around X  J/L=CW/ACW around Y  U/O=CW/ACW around Z\n"
                "  Gripper:     G (toggle)   Speed: +/- keys"
            )
        else:
            logger.info("  Freedrive — guide robot by hand.  G = gripper toggle.")

    def get_action(self) -> dict[str, float]:
        """Return target joint positions + gripper."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        q_now = np.array(self.rtde_rec.getActualQ())  # current 6 joints

        # ── Cartesian keyboard mode ────────────────────────
        if self.config.mode == "cartesian":
            now = time.perf_counter()
            dt = min(now - self._last_time, 0.1)     # cap to avoid jumps
            self._last_time = now

            # Accumulate Cartesian delta from pressed keys
            dx = np.zeros(6)
            with self._lock:
                pressed = set(self._pressed)          # snapshot
                speed_mult = self._speed_mult
            for key, (axis, sign) in self._KEY_MAP.items():
                if key in pressed:
                    vel = self.config.trans_vel if axis < 3 else self.config.rot_vel
                    dx[axis] += sign * vel * speed_mult * dt

            # Jacobian IK: Cartesian delta → joint delta
            if np.linalg.norm(dx) > 1e-10:
                dq = cartesian_to_joint_delta(q_now, dx)
                dq = np.clip(dq, -0.05, 0.05)        # safety clamp ~3°/step
                q_target = q_now + dq
            else:
                q_target = q_now                       # hold position

            # Periodic status print
            if time.time() - self._last_log > 3.0:
                tcp = self.rtde_rec.getActualTCPPose()
                g_str = "CLOSED" if self._gripper > 0.5 else "OPEN"
                logger.info(
                    f"TCP [{tcp[0]:.3f} {tcp[1]:.3f} {tcp[2]:.3f}] "
                    f"speed={speed_mult:.1f}x  gripper={g_str}"
                )
                self._last_log = time.time()

        # ── Freedrive mode ─────────────────────────────────
        else:
            q_target = q_now                           # just echo current joints

        with self._lock:
            gripper = self._gripper

        result = {f"joint_{i}": float(q_target[i]) for i in range(6)}
        result["gripper"] = gripper
        return result

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self.rtde_rec.disconnect()
        self.rtde_rec = None
        logger.info(f"{self} disconnected")

    # ── Keyboard listener ──────────────────────────────────

    def _start_keyboard(self) -> None:
        try:
            from pynput import keyboard

            def _on_press(key):
                with self._lock:
                    # Character keys (wasd, ijkl, etc.)
                    try:
                        if key.char:
                            c = key.char.lower()
                            self._pressed.add(c)
                            if c in ("+", "="):
                                self._speed_mult = min(self._speed_mult + 0.25, 3.0)
                                logger.info(f"Speed → {self._speed_mult:.2f}x")
                            elif c in ("-", "_"):
                                self._speed_mult = max(self._speed_mult - 0.25, 0.25)
                                logger.info(f"Speed → {self._speed_mult:.2f}x")
                    except AttributeError:
                        pass
                    # Gripper toggle on 'G' key
                    try:
                        if key.char and key.char.lower() == 'g':
                            self._gripper = 0.0 if self._gripper > 0.5 else 1.0
                            g_str = "CLOSED" if self._gripper > 0.5 else "OPEN"
                            logger.info(f"Gripper → {g_str}")
                    except AttributeError:
                        pass

            def _on_release(key):
                with self._lock:
                    try:
                        if key.char:
                            self._pressed.discard(key.char.lower())
                    except AttributeError:
                        pass

            self._listener = keyboard.Listener(
                on_press=_on_press, on_release=_on_release
            )
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            logger.warning(f"Could not start keyboard listener: {e}")