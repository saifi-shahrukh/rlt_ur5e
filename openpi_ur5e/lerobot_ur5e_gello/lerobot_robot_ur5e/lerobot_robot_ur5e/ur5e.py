"""UR5e robot interface — with freedrive mode support."""

from typing import Optional, Any
import logging
import numpy as np
import rtde_control
import rtde_receive
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceNotConnectedError
from .robotiq_gripper import RobotiqGripper
from .config_ur5e import UR5EConfig, UR5EDualCamConfig

logger = logging.getLogger(__name__)


class UR5E(Robot):
    config_class = UR5EConfig
    name = "ur5e"

    def __init__(self, config: UR5EConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.robot_ip = config.ip
        self.freedrive = config.freedrive
        self.calibrate_gripper = getattr(config, 'calibrate_gripper', False)
        self.rtde_ctrl: Optional[rtde_control.RTDEControlInterface] = None
        self.rtde_rec: Optional[rtde_receive.RTDEReceiveInterface] = None

        self.acc = 0.5
        self.speed = 0.5
        self.servoj_t = 1.0 / 500
        self.servoj_lookahead = 0.2
        self.servoj_gain = 100

        self.with_gripper = True
        self.gripper = RobotiqGripper()
        self.gripper_speed = 255
        self.gripper_force = 10

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_0": float, "joint_1": float, "joint_2": float,
            "joint_3": float, "joint_4": float, "joint_5": float,
            "gripper": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.rtde_ctrl is not None
            and self.rtde_rec is not None
            and self.rtde_ctrl.isConnected()
            and self.rtde_rec.isConnected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = None) -> None:
        """Connect to UR5e robot.
        
        Args:
            calibrate: Whether to calibrate gripper. If None, uses config value.
                      Set False to skip calibration when peg is already grasped.
        """
        if self.is_connected:
            return
        
        if calibrate is None:
            calibrate = self.calibrate_gripper

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connecting to UR5e at {self.robot_ip} (attempt {attempt}/{max_retries})...")
                self.rtde_ctrl = rtde_control.RTDEControlInterface(self.robot_ip)
                self.rtde_rec = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                self.gripper.connect(self.robot_ip, 63352)
                self.gripper.activate(auto_calibrate=calibrate)
                logger.info(f"✓ UR5e connected successfully!")
                break
            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {e}")
                self.rtde_ctrl = None
                self.rtde_rec = None
                if attempt < max_retries:
                    logger.info("  Retrying in 3 seconds...")
                    logger.info("  ► If robot has protective stop: clear it on teach pendant")
                    logger.info("  ► Make sure robot is in Remote Control mode")
                    import time
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"\n\n{'='*60}\n"
                        f"  FAILED to connect to UR5e after {max_retries} attempts!\n"
                        f"  Error: {e}\n\n"
                        f"  CHECKLIST:\n"
                        f"  1. Clear any Protective Stop on teach pendant\n"
                        f"  2. Ensure robot is in 'Remote Control' mode\n"
                        f"  3. Check robot IP: ping {self.robot_ip}\n"
                        f"  4. Restart robot program if needed\n"
                        f"{'='*60}\n"
                    ) from e

        for cam in self.cameras.values():
            cam.connect()

        if self.freedrive:
            self.rtde_ctrl.teachMode()
            logger.info("FREEDRIVE (teach-mode) ON")

        self.configure()

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self.freedrive and self.rtde_ctrl and self.rtde_ctrl.isConnected():
            self.rtde_ctrl.endTeachMode()
            logger.info("FREEDRIVE (teach-mode) OFF")
        if self.rtde_ctrl:
            self.rtde_ctrl.disconnect()
        if self.rtde_rec:
            self.rtde_rec.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    def recover_from_protective_stop(self) -> bool:
        """Attempt to recover from a protective stop or RTDE failure.

        Uses the correct UR5e recovery procedure based on ur_rtde documentation:
        1. Disconnect broken RTDE interfaces
        2. Wait minimum 5 seconds (UR5e hardware requirement for protective stop unlock)
        3. Use DashboardClient to programmatically unlock the protective stop
        4. Re-establish fresh RTDE connections with script reupload
        5. Move robot slightly away from the failure point (5mm up)

        Returns:
            True if recovery succeeded, False otherwise.

        Usage during inference:
            try:
                robot.send_action(action)
            except RuntimeError as e:
                if "RTDE" in str(e) or "protective" in str(e).lower():
                    if robot.recover_from_protective_stop():
                        continue  # Robot recovered, keep going
                    else:
                        break  # Recovery failed, stop
        """
        import time
        import dashboard_client

        logger.warning("⚠️  Attempting recovery from protective stop / RTDE failure...")

        # Step 1: Disconnect broken RTDE interfaces
        try:
            if self.rtde_ctrl:
                self.rtde_ctrl.disconnect()
            if self.rtde_rec:
                self.rtde_rec.disconnect()
        except Exception:
            pass  # May already be broken

        self.rtde_ctrl = None
        self.rtde_rec = None

        # Step 2: Wait 5 seconds (UR5e REQUIRES minimum 5s before unlock)
        logger.info("  Waiting 5s (UR5e requires 5s before protective stop can be unlocked)...")
        time.sleep(5.0)

        # Step 3: Use Dashboard Client to unlock the protective stop
        try:
            logger.info("  Connecting to Dashboard (port 29999) to unlock protective stop...")
            dash = dashboard_client.DashboardClient(self.robot_ip)
            dash.connect()

            # Check current safety mode
            safety_mode = dash.safetymode()
            logger.info(f"  Safety mode: {safety_mode}")

            # Unlock the protective stop
            dash.unlockProtectiveStop()
            logger.info("  ✓ Protective stop unlocked via Dashboard!")
            time.sleep(0.5)

            # Close any safety popups on pendant
            try:
                dash.closeSafetyPopup()
            except Exception:
                pass

            dash.disconnect()
        except Exception as e:
            logger.warning(f"  Dashboard unlock failed: {e}")
            logger.info("  ► You may need to manually clear on teach pendant")
            # Continue anyway — sometimes the stop auto-clears

        # Step 4: Re-establish fresh RTDE connections with script reupload
        for attempt in range(1, 11):
            try:
                logger.info(f"  Reconnecting RTDE (attempt {attempt}/10)...")
                self.rtde_ctrl = rtde_control.RTDEControlInterface(self.robot_ip)
                self.rtde_rec = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                logger.info(f"  ✓ RTDE reconnected on attempt {attempt}!")
                break
            except Exception as e:
                logger.warning(f"  Attempt {attempt} failed: {e}")
                self.rtde_ctrl = None
                self.rtde_rec = None
                if attempt < 10:
                    time.sleep(1.0)
        else:
            logger.error("  ✗ Failed to reconnect after 10 attempts. Manual intervention needed.")
            logger.error("  ► Clear protective stop on teach pendant, then restart script.")
            return False

        # Step 5: Reupload RTDE control script (in case it was lost)
        try:
            success = self.rtde_ctrl.reuploadScript()
            if success:
                logger.info("  ✓ RTDE control script reuploaded.")
            else:
                logger.info("  Script already running (reupload not needed).")
        except Exception as e:
            logger.warning(f"  Script reupload note: {e}")

        # Step 6: Move robot slightly away from failure point (5mm upward in Z)
        try:
            current_pose = self.rtde_rec.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            retreat_pose = list(current_pose)
            retreat_pose[2] += 0.005  # Move 5mm up in Z
            logger.info(f"  Moving 5mm up from failure point (z: {current_pose[2]:.4f} → {retreat_pose[2]:.4f})")
            self.rtde_ctrl.moveL(retreat_pose, 0.1, 0.1)  # Slow, safe retreat
            time.sleep(0.5)
            logger.info("  ✓ Recovery complete! Robot moved to safe position.")
            return True
        except Exception as e:
            logger.error(f"  ✗ Failed to retreat from failure point: {e}")
            logger.error("  Robot may still be usable — try sending new actions.")
            return True  # Connection is restored even if move fails

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        joint_positions = self.rtde_rec.getActualQ()
        gripper_position = self.gripper.get_current_position() / 255.0
        obs_dict = {f"joint_{i}": val for i, val in enumerate(joint_positions)}
        obs_dict["gripper"] = gripper_position
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not all(key in self.action_features for key in action.keys()):
            raise ValueError(f"Invalid action keys: {set(action) - set(self.action_features)}")
        goal_gripper = int(np.clip(action["gripper"] * 255.0, 0, 255))
        if self.freedrive:
            self.gripper.move(goal_gripper, self.gripper_speed, self.gripper_force)
        else:
            goal_joints = [action[f"joint_{i}"] for i in range(6)]
            self.rtde_ctrl.servoJ(
                goal_joints, self.acc, self.speed,
                self.servoj_t, self.servoj_lookahead, self.servoj_gain,
            )
            self.gripper.move(goal_gripper, self.gripper_speed, self.gripper_force)
        return action


class UR5EDualCam(UR5E):
    """UR5E with dual cameras (RealSense wrist + Kinect/OpenCV overhead).

    Identical to UR5E but uses UR5EDualCamConfig which includes the overhead camera.
    """
    config_class = UR5EDualCamConfig
    name = "ur5e_dual_cam"

    def __init__(self, config: UR5EDualCamConfig):
        super().__init__(config)