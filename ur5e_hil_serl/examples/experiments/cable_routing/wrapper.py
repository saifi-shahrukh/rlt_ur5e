"""Cable routing environment for UR5e + Hand-E.

The robot must route a cable through a series of clips.
Gripper stays closed (cable is pre-grasped).
Larger workspace and rotation range than insertion tasks.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from ur_env.envs.ur5e_env import UR5eEnv


class CableRoutingEnv(UR5eEnv):
    """Cable routing task for UR5e.

    The robot holds a cable and must route it through clips.
    Requires larger motions and more rotation than insertion tasks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._first_reset = True

    def go_to_reset(self, joint_reset=False):
        """
        Custom reset for cable routing:
        1. HOME_Q on first reset (safe start)
        2. Move to reset joint configuration
        3. Optional random perturbation (larger range)
        """
        # On first reset OR explicit joint_reset: go to HOME_Q first
        if self._first_reset or joint_reset:
            print("[CableRoute] Moving to HOME_Q first (safe position)")
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

        # Ensure gripper closed (cable stays grasped)
        self.controller.close_gripper()
        time.sleep(0.2)

        # Random perturbation (larger for cable routing)
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

        # Ensure gripper closed (cable stays grasped)
        self.controller.close_gripper()
        time.sleep(0.2)

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}
