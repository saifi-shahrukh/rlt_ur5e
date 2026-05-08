"""Peg insertion environment for UR5e + Hand-E.

The reset is now handled properly inside the controller thread:
- Controller uses speedL/speedJ to retract (avoids singularity)
- Then moveJ to reset joints
- Then restarts force mode

This matches the proven ur5e_serl architecture where the controller
thread owns the entire reset sequence.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from ur_env.envs.ur5e_env import UR5eEnv


class PegInsertionEnv(UR5eEnv):
    """Peg insertion task for UR5e.

    The peg is pre-grasped. Episodes start with the peg above the hole.
    The policy must insert the peg into the hole.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
        self._first_reset = True  # Go to HOME_Q on first reset

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
        Reset for peg insertion.
        
        Sequence:
        1. Gently retract peg upward (via impedance target)
        2. HOME_Q on first reset (safe start from any position)
        3. moveJ to RESET_Q
        4. Set curr_reset_pose for MRP safety clipping
        """
        # Step 1: Gently retract peg upward before any joint move
        # This prevents jamming when peg is inserted in the hole
        if self.controller is not None and self.curr_path_length > 0:
            self._update_currpos()
            retract_pose = self.currpos.copy()
            retract_pose[2] = max(retract_pose[2] + 0.08, 0.15)
            self._send_pos_command(retract_pose)
            for _ in range(30):  # wait up to 3 seconds
                time.sleep(0.1)
                self._update_currpos()
                if self.currpos[2] > retract_pose[2] - 0.01:
                    break

        # Step 2: On first reset OR explicit joint_reset: go to HOME_Q first
        if self._first_reset or joint_reset:
            self.controller.close_gripper()
            time.sleep(0.3)
            print("[PegInsert] Moving to HOME_Q first (safe position)")
            self.controller.move_to_joints(self.config.HOME_Q)
            time.sleep(0.3)
            self._first_reset = False

        # Step 3: Move to task start joints
        reset_q = self._reset_Q[0] if self._reset_Q.ndim == 2 else self._reset_Q
        self.controller.move_to_joints(reset_q)
        time.sleep(0.2)

        # Update state after reset move
        self._update_currpos()

        # Store reset pose for MRP-based safety clipping
        self.curr_reset_pose[:] = self.currpos.copy()

        # Ensure gripper closed (peg must stay grasped)
        self.controller.close_gripper()
        time.sleep(0.2)

        # Random perturbation on top of reset pose
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
            self._update_currpos()

    def regrasp(self):
        """Re-grasp the peg if it was dropped (triggered by F1)."""
        print("[PegInsert] RE-GRASP triggered")

        # Open gripper
        input("Press ENTER to open gripper...")
        self.controller.open_gripper()
        time.sleep(0.5)

        # Wait for user to place peg
        input("Place peg in gripper and press ENTER to close...")

        # Close gripper
        self.controller.close_gripper()
        time.sleep(1.0)

    def reset(self, joint_reset=False, **kwargs):
        """Reset with retract handled by controller."""
        self.last_gripper_act = time.time()

        if self.save_video:
            self.save_video_recording()

        # Cycle count for periodic joint resets
        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            joint_reset = True

        # Handle re-grasp request
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        # Main reset sequence
        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        # Ensure gripper is closed (peg must stay gripped)
        if self.controller is not None:
            for _ in range(3):
                self.controller.close_gripper()
                time.sleep(0.2)

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}
