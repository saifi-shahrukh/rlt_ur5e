"""Bin relocation (pick-place) environment for UR5e + Hand-E.

Mirrors hil-serl's usb_pickup_insertion but for our UR5e setup.
This task involves picking an object and placing it in a target bin.
Gripper must be learned (open/close during episode).
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from ur_env.envs.ur5e_env import UR5eEnv
from ur_env.utils.rotations import euler_2_quat


class BinRelocationEnv(UR5eEnv):
    """Pick-place task for UR5e.

    The robot must:
    1. Pick up an object from a source location
    2. Move it to a target bin
    3. Release it

    Gripper control is learned (not fixed).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._first_reset = True

    def go_to_reset(self, joint_reset=False):
        """
        Custom reset for bin relocation:
        1. HOME_Q on first reset (safe start)
        2. Open gripper to release any held object
        3. Move to reset joint configuration
        4. Optional random perturbation
        """
        # On first reset OR explicit joint_reset: go to HOME_Q first
        if self._first_reset or joint_reset:
            print("[BinReloc] Moving to HOME_Q first (safe position)")
            self.controller.move_to_joints(self.config.HOME_Q)
            time.sleep(0.3)
            self._first_reset = False

        # Open gripper (release object for pick-place task)
        self.controller.open_gripper()
        time.sleep(0.4)

        # Move to task reset joints
        reset_q = self._reset_Q[0] if self._reset_Q.ndim == 2 else self._reset_Q
        self.controller.move_to_joints(reset_q)
        time.sleep(0.3)

        self._update_currpos()
        # Store reset pose for MRP-based safety clipping
        self.curr_reset_pose[:] = self.currpos.copy()

        # Random perturbation
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

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            joint_reset = True

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}
