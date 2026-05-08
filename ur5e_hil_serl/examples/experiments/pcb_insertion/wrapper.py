"""PCB insertion environment for UR5e + Hand-E.

Similar to peg insertion but with tighter tolerances.
The PCB connector is pre-grasped, gripper stays closed.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from ur_env.envs.ur5e_env import UR5eEnv


class PCBInsertionEnv(UR5eEnv):
    """PCB insertion task for UR5e.

    Pre-grasped PCB connector must be inserted into a socket.
    Tighter tolerances than peg insertion.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
        self._first_reset = True

        try:
            from pynput import keyboard

            def on_press(key):
                if str(key) == "Key.f1":
                    self.should_regrasp = True

            self._regrasp_listener = keyboard.Listener(on_press=on_press)
            self._regrasp_listener.start()
        except ImportError:
            pass

    def go_to_reset(self, joint_reset=False):
        """
        Custom reset for PCB insertion:
        1. HOME_Q on first reset (safe start)
        2. Retract gently (PCB connectors are fragile)
        3. Move to reset joint configuration
        4. Optional random XY perturbation
        """
        # On first reset OR explicit joint_reset: go to HOME_Q first
        if self._first_reset or joint_reset:
            print("[PCBInsert] Moving to HOME_Q first (safe position)")
            self.controller.move_to_joints(self.config.HOME_Q)
            time.sleep(0.3)
            self._first_reset = False

        # Move to task reset joints
        reset_q = self._reset_Q[0] if self._reset_Q.ndim == 2 else self._reset_Q
        self.controller.move_to_joints(reset_q)
        time.sleep(0.3)

        self._update_currpos()
        # Store reset pose for MRP-based safety clipping
        self.curr_reset_pose[:] = self.currpos.copy()

        # Ensure gripper closed (PCB must stay grasped)
        self.controller.close_gripper()
        time.sleep(0.2)

        # Random perturbation (smaller than peg insertion)
        if self.randomreset:
            reset_pose = self.currpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler = R.from_quat(reset_pose[3:]).as_euler("xyz")
            euler[2] += np.random.uniform(-self.random_rz_range, self.random_rz_range)
            reset_pose[3:] = R.from_euler("xyz", euler).as_quat()
            self._send_pos_command(reset_pose)
            time.sleep(0.5)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()

        if self.save_video:
            self.save_video_recording()

        if self.should_regrasp:
            # Simple regrasp: open, wait for user, close
            print("[PCBInsert] RE-GRASP")
            self._update_currpos()
            retract_pose = self.currpos.copy()
            retract_pose[2] += 0.05
            self.interpolate_move(retract_pose, timeout=0.8)
            self.controller.open_gripper()
            input("Place PCB and press enter...")
            self.controller.close_gripper()
            time.sleep(1.0)
            self.should_regrasp = False

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        # Ensure gripper closed
        self.controller.close_gripper()
        time.sleep(0.2)

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}
