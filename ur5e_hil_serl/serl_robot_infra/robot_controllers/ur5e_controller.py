import datetime
import time
import threading
import asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur_env.utils.hande_gripper import HandEGripper
from ur_env.utils.rotations import rotvec_2_quat, quat_2_rotvec, pose_2_rotvec, pose_2_quat

np.set_printoptions(precision=4, suppress=True)


def pos_difference(quat_pose_1: np.ndarray, quat_pose_2: np.ndarray):
    assert quat_pose_1.shape == (7,) and quat_pose_2.shape == (7,)
    p_diff = np.sum(np.abs(quat_pose_1[:3] - quat_pose_2[:3]))
    r_diff = (R.from_quat(quat_pose_1[3:]) * R.from_quat(quat_pose_2[3:]).inv()).magnitude()
    return p_diff + r_diff


class UrImpedanceController(threading.Thread):
    """
    Impedance controller for UR5e with Hand-E gripper.
    
    Supports both vacuum-style tasks (gripper releases on reset)
    and peg-insertion tasks (gripper stays closed on reset).
    
    Config flags:
      GRIPPER_RELEASE_ON_RESET: bool (default True)
        True  → gripper opens before reset move (pick-place tasks)
        False → gripper stays closed during reset (peg insertion)
      HOME_Q: np.ndarray (optional)
        Safe joint position for cleanup on disconnect.
        If not set, uses a sensible default.
    """

    # Safe home position: deg2rad([45, -68, -102, -100, 90, 0])
    DEFAULT_HOME_Q = np.array(
        [0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0],
        dtype=np.float32,
    )

    def __init__(
            self,
            robot_ip,
            frequency=100,
            kp=10000,
            kd=2200,
            config=None,
            verbose=False,
            plot=False,
            *args,
            **kwargs
    ):
        super(UrImpedanceController, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._is_ready = threading.Event()
        self._is_truncated = threading.Event()
        self.lock = threading.Lock()

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.kp = kp
        self.kd = kd
        self.verbose = verbose
        self.do_plot = plot

        self.target_pos = np.zeros((7,), dtype=np.float32)
        self.target_grip = np.zeros((1,), dtype=np.float32)
        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force_lowpass = np.zeros((6,), dtype=np.float32)
        self.curr_force = np.zeros((6,), dtype=np.float32)
        self.curr_timestamp_ms = 0

        # Task reset position (set by env via set_reset_Q / set_reset_angles)
        self.reset_Q = np.array(
            [np.pi / 2., -np.pi / 2., np.pi / 2., -np.pi / 2., -np.pi / 2., 0.],
            dtype=np.float32,
        )
        self.reset_Pose = np.zeros(6, dtype=np.float32)
        self.reset_height = np.array([0.1], dtype=np.float32)

        # Safe home for cleanup — configurable or default
        self.home_Q = getattr(config, "HOME_Q", self.DEFAULT_HOME_Q).copy()

        # Whether to release gripper during reset (True for pick-place, False for peg)
        self.gripper_release_on_reset = getattr(config, "GRIPPER_RELEASE_ON_RESET", True)

        # Impedance / force-mode parameters
        self.delta = config.ERROR_DELTA
        self.fm_damping = config.FORCEMODE_DAMPING
        self.fm_task_frame = config.FORCEMODE_TASK_FRAME
        self.fm_selection_vector = config.FORCEMODE_SELECTION_VECTOR
        self.fm_limits = config.FORCEMODE_LIMITS

        self.ur_control: RTDEControlInterface = None
        self.ur_receive: RTDEReceiveInterface = None
        self.gripper: HandEGripper = None

        self.err = 0
        self.noerr = 0

        with open("/tmp/console2.txt", 'w') as f:
            f.write("reset\n")
        self.second_console = open("/tmp/console2.txt", 'a')

    def start(self):
        super().start()
        if self.verbose:
            print(f"[RIC] Controller process spawned at {self.native_id}")

    def print(self, msg, both=False):
        self.second_console.write(f'{datetime.datetime.now()} --> {msg}\n')
        if both:
            print(msg)

    async def start_ur_interfaces(self, gripper=True):
        self.ur_control = RTDEControlInterface(self.robot_ip)
        self.ur_receive = RTDEReceiveInterface(self.robot_ip)
        if gripper:
            self.gripper = HandEGripper(self.robot_ip)
            await self.gripper.connect()
            await self.gripper.activate()
        if self.verbose:
            gr_string = "(with Hand-E) " if gripper else ""
            print(f"[RIC] Controller connected to robot {gr_string}at: {self.robot_ip}")

    async def restart_ur_interface(self):
        self._is_truncated.set()
        self.print("[RIC] forcemode failed, is now truncated!", both=True)
        self.ur_control.disconnect()
        try:
            self.ur_control.reconnect()
        except RuntimeError:
            self.ur_receive.disconnect()
            for _ in range(10):
                try:
                    self.ur_control.disconnect()
                    self.ur_receive.disconnect()
                    await self.start_ur_interfaces(gripper=False)
                    return
                except Exception as e:
                    print(e)
                    time.sleep(0.2)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def is_moving(self):
        return np.linalg.norm(self.get_state()["vel"], 2) > 0.01

    def set_target_pos(self, target_pos: np.ndarray):
        if target_pos.shape == (7,):
            target_orientation = target_pos[3:]
        elif target_pos.shape == (6,):
            target_orientation = rotvec_2_quat(target_pos[3:])
        else:
            raise ValueError(f"[RIC] target pos has shape {target_pos.shape}")
        with self.lock:
            self.target_pos[:3] = target_pos[:3]
            self.target_pos[3:] = target_orientation
            self.print(f"target: {self.target_pos}")

    def set_reset_Q(self, reset_Q: np.ndarray):
        with self.lock:
            self.reset_Q[:] = reset_Q
        self._reset.set()

    def set_reset_pose(self, reset_pose: np.ndarray):
        with self.lock:
            self.reset_Pose[:] = reset_pose
        self._reset.set()

    def set_gripper_pos(self, target_grip: np.ndarray):
        with self.lock:
            self.target_grip[:] = target_grip

    def get_target_pos(self, copy=True):
        with self.lock:
            return self.target_pos.copy() if copy else self.target_pos

    async def _update_robot_state(self):
        pos = self.ur_receive.getActualTCPPose()
        vel = self.ur_receive.getActualTCPSpeed()
        Q = self.ur_receive.getActualQ()
        Qd = self.ur_receive.getActualQd()
        force = self.ur_receive.getActualTCPForce()

        # Hand-E state
        grip_pos = await self.gripper.get_position_normalized()
        obj_status = await self.gripper.get_object_status()
        obj_detected = float(obj_status in (
            HandEGripper.ObjectStatus.CONTACT_OPENING,
            HandEGripper.ObjectStatus.CONTACT_CLOSING,
        ))

        with self.lock:
            self.curr_pos[:] = pose_2_quat(pos)
            self.curr_vel[:] = vel
            self.curr_Q[:] = Q
            self.curr_Qd[:] = Qd
            self.curr_force[:] = np.array(force)
            self.curr_force_lowpass[:] = (
                0.1 * np.array(force) + 0.9 * self.curr_force_lowpass
            )
            self.gripper_state[:] = [grip_pos, obj_detected]
            self.curr_timestamp_ms = int(time.monotonic() * 1000)

    def get_state(self):
        with self.lock:
            return {
                "pos": self.curr_pos.copy(),
                "vel": self.curr_vel.copy(),
                "Q": self.curr_Q.copy(),
                "Qd": self.curr_Qd.copy(),
                "force": self.curr_force_lowpass[:3].copy(),
                "torque": self.curr_force_lowpass[3:].copy(),
                "gripper": self.gripper_state.copy(),
                "timestamp_ms": self.curr_timestamp_ms,
            }

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    def _calculate_force(self):
        target_pos = self.get_target_pos(copy=True)
        with self.lock:
            curr_pos = self.curr_pos
            curr_vel = self.curr_vel

        kp, kd = self.kp, self.kd
        diff_p = np.clip(
            target_pos[:3] - curr_pos[:3], -self.delta, self.delta
        )
        vel_delta = 2 * self.delta * self.frequency
        diff_d = np.clip(-curr_vel[:3], -vel_delta, vel_delta)
        force_pos = kp * diff_p + kd * diff_d

        rot_diff = R.from_quat(target_pos[3:]) * R.from_quat(curr_pos[3:]).inv()
        vel_rot_diff = R.from_rotvec(curr_vel[3:]).inv()
        torque = rot_diff.as_rotvec() * 100 + vel_rot_diff.as_rotvec() * 22

        # Adaptive downward force limiting
        if self.curr_force[2] > 3.5 and force_pos[2] < 0.0:
            force_pos[2] = (
                max(1.5 - self.curr_force_lowpass[2], 0.0) * force_pos[2]
                + min(self.curr_force_lowpass[2] - 0.5, 1.0) * 20.0

            )

        return np.concatenate((force_pos, torque))

    async def send_gripper_command(self, force_release=False):
        """Hand-E gripper: >0.5 close, <-0.5 open, neutral=no action."""
        if force_release:
            await self.gripper.open(speed=255, force=100)
            self.target_grip[0] = 0.0
            return

        tgt = self.target_grip[0]
        if tgt > 0.5:
            await self.gripper.close(speed=255, force=150)
        elif tgt < -0.5:
            await self.gripper.open(speed=255, force=100)
        # else: neutral (no command) — gripper holds current position

    def _truncate_check(self):
        downward_force = self.curr_force_lowpass[2] > 20.0
        if downward_force:
            self._is_truncated.set()
        else:
            self._is_truncated.clear()

    def is_truncated(self):
        return self._is_truncated.is_set()

    # Aliases for ur5_env.py compatibility
    def set_target_pose(self, target_pose: np.ndarray):
        self.set_target_pos(target_pose)

    def set_reset_angles(self, reset_Q: np.ndarray):
        self.set_reset_Q(reset_Q)

    def run(self):
        try:
            asyncio.run(self.run_async())
        finally:
            self.stop()

    async def _go_to_reset_pose(self):
        self.ur_control.forceModeStop()

        # Gripper behavior on reset — configurable per task
        if self.gripper and self.gripper_release_on_reset:
            await self.send_gripper_command(force_release=True)
        time.sleep(0.01)

        # Move up to avoid dragging
        success = True
        while self.curr_pos[2] < self.reset_height:
            if self.curr_Q[2] < 0.5:
                success = success and self.ur_control.speedJ(
                    [0.0, -1.0, 1.0, 0.0, 0.0, 0.0], acceleration=0.8
                )
            else:
                success = success and self.ur_control.speedL(
                    [0.0, 0.0, 0.25, 0.0, 0.0, 0.0], acceleration=0.8
                )
            await self._update_robot_state()
            time.sleep(0.01)
        self.ur_control.speedStop(a=1.0)

        if self.reset_Pose.std() > 0.001:
            success = success and self.ur_control.moveL(
                self.reset_Pose.tolist(), speed=0.5, acceleration=0.3
            )
            self.reset_Pose[:] = 0.0
        else:
            success = success and self.ur_control.moveJ(
                self.reset_Q.tolist(), speed=1.0, acceleration=0.8
            )

        time.sleep(0.1)
        await self._update_robot_state()
        with self.lock:
            self.target_pos = self.curr_pos.copy()

        self.ur_control.forceModeSetDamping(self.fm_damping)
        self.ur_control.zeroFtSensor()

        if not success:
            await self.restart_ur_interface()
        else:
            self._is_truncated.clear()
            self._reset.clear()

    async def run_async(self):
        await self.start_ur_interfaces(gripper=True)
        self.ur_control.forceModeSetDamping(self.fm_damping)

        try:
            dt = 1.0 / self.frequency
            self.ur_control.zeroFtSensor()
            await self._update_robot_state()
            self.target_pos = self.curr_pos.copy()
            print(f"[RIC] target position set to curr pos: {self.target_pos}")
            self._is_ready.set()

            while not self.stopped():
                if self._reset.is_set():
                    await self._update_robot_state()
                    await self._go_to_reset_pose()

                t_now = time.monotonic()
                await self._update_robot_state()
                self._truncate_check()

                force = self._calculate_force()
                self.print(
                    f" p:{self.curr_pos}   f:{self.curr_force_lowpass}"
                    f"   gr:{self.gripper_state}"
                )

                t_start = self.ur_control.initPeriod()
                fm_successful = self.ur_control.forceMode(
                    self.fm_task_frame.tolist(),
                    self.fm_selection_vector.tolist(),
                    force.tolist(),
                    2,
                    self.fm_limits.tolist(),
                )
                if not fm_successful:
                    self.print("[RIC] forceMode failed, recovering...", both=True)
                    await self.restart_ur_interface()
                    await self._go_to_reset_pose()

                if self.gripper:
                    await self.send_gripper_command()

                self.ur_control.waitPeriod(t_start)

                a = dt - (time.monotonic() - t_now)
                time.sleep(max(0.0, a))
                self.err += int(a < 0.0)
                self.noerr += int(a >= 0.0)
                if a < -0.04:
                    self.print(
                        f"Controller stopped for "
                        f"{(time.monotonic() - t_now) * 1e3:.1f} ms"
                    )

        finally:
            if self.verbose:
                print(f"[RIC] >dt: {self.err}  <dt (good): {self.noerr}")
            self.ur_control.forceModeStop()

            # Release gripper on shutdown only if configured to do so
            if self.gripper and self.gripper_release_on_reset:
                await self.send_gripper_command(force_release=True)
                time.sleep(0.05)

            # Move to safe home position
            self.ur_control.moveJ(
                self.home_Q.tolist(), speed=1.0, acceleration=0.8
            )
            self.ur_control.disconnect()
            self.ur_receive.disconnect()
            if self.verbose:
                print(f"[RIC] Disconnected from: {self.robot_ip}")
    # ==================================================================
    # INTERFACE ALIASES for ur5e_hil_serl's UR5eEnv compatibility
    # ==================================================================

    def move_to_joints(self, q: np.ndarray, speed=1.0, acceleration=0.8):
        """Blocking joint move — triggers reset sequence in control loop."""
        with self.lock:
            self.reset_Q[:] = np.array(q, dtype=np.float32)
        self._reset.set()
        # Wait for reset to complete
        timeout = 15.0
        start = time.time()
        while self._reset.is_set():
            time.sleep(0.05)
            if time.time() - start > timeout:
                print("[RIC] WARNING: move_to_joints timeout!")
                break

    def open_gripper(self):
        """Open gripper (non-blocking)."""
        with self.lock:
            self.target_grip[:] = -1.0

    def close_gripper(self):
        """Close gripper (non-blocking)."""
        with self.lock:
            self.target_grip[:] = 1.0
