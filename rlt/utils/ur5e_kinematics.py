"""UR5e Kinematics Utilities — Joint ↔ Cartesian conversion.

Converts between joint-space actions (VLA/RLT output) and Cartesian-space
actions (SERL impedance controller input).

Two modes:
  1. RTDE live: Use the robot's built-in getJacobian() (most accurate)
  2. Offline: Use UR5e DH parameters for analytical Jacobian (no robot needed)

Usage:
    from rlt.utils.ur5e_kinematics import UR5eKinematics
    
    kin = UR5eKinematics(mode="dh")  # or mode="rtde" with controller
    
    # Joint delta → Cartesian delta
    dq = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.005])  # 6 joint deltas (rad)
    q_current = np.array([0.6, -1.35, -2.31, -1.03, 1.54, 0.65])  # current joints
    dx = kin.joint_to_cartesian(dq, q_current)  # (6,) [x,y,z,rx,ry,rz] meters/rad
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


# UR5e DH Parameters (modified DH convention)
# From Universal Robots documentation
UR5E_DH = {
    "a": [0, -0.42500, -0.39225, 0, 0, 0],
    "d": [0.1625, 0, 0, 0.1333, 0.0997, 0.0996],
    "alpha": [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
}


def _dh_transform(theta, d, a, alpha):
    """Compute 4x4 homogeneous transform from DH parameters."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])


def _forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Compute FK for UR5e (returns 4x4 end-effector transform)."""
    a = UR5E_DH["a"]
    d = UR5E_DH["d"]
    alpha = UR5E_DH["alpha"]
    
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_transform(q[i], d[i], a[i], alpha[i])
    return T


def _numerical_jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute 6x6 geometric Jacobian numerically.
    
    Returns:
        J: (6, 6) matrix where dx = J @ dq
           Top 3 rows: linear velocity (m/s per rad/s)
           Bottom 3 rows: angular velocity (rad/s per rad/s)
    """
    J = np.zeros((6, 6))
    T0 = _forward_kinematics(q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]
    
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += eps
        T_plus = _forward_kinematics(q_plus)
        
        # Linear part (position difference)
        J[:3, i] = (T_plus[:3, 3] - p0) / eps
        
        # Angular part (rotation difference as rotation vector)
        dR = T_plus[:3, :3] @ R0.T
        # Convert small rotation matrix to rotation vector
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if angle < 1e-10:
            J[3:, i] = np.zeros(3)
        else:
            # Use scipy for robustness
            rv = R.from_matrix(dR).as_rotvec()
            J[3:, i] = rv / eps
    
    return J


class UR5eKinematics:
    """Joint ↔ Cartesian conversion for UR5e.
    
    Supports two modes:
    - "dh": Use analytical DH parameters (works offline)
    - "rtde": Use robot's built-in getJacobian (requires live connection)
    """
    
    def __init__(self, mode: str = "dh", rtde_controller=None):
        """
        Args:
            mode: "dh" for DH-based computation, "rtde" for live robot
            rtde_controller: RTDEControlInterface instance (only for mode="rtde")
        """
        self.mode = mode
        self.rtde_controller = rtde_controller
        
        if mode == "rtde" and rtde_controller is None:
            print("[UR5eKinematics] WARNING: rtde mode but no controller. Falling back to DH.")
            self.mode = "dh"
    
    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Get 6x6 Jacobian at joint configuration q.
        
        Args:
            q: (6,) joint angles in radians
            
        Returns:
            J: (6, 6) Jacobian matrix
        """
        if self.mode == "rtde" and self.rtde_controller is not None:
            try:
                # RTDE returns flat list of 36 elements (6x6 row-major)
                j_flat = self.rtde_controller.getJacobian(list(q))
                return np.array(j_flat).reshape(6, 6)
            except Exception as e:
                # Fallback to DH
                pass
        
        return _numerical_jacobian(q)
    
    def joint_to_cartesian(self, dq: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Convert joint-space delta to Cartesian-space delta.
        
        Args:
            dq: (6,) joint angle deltas in radians
            q: (6,) current joint angles in radians
            
        Returns:
            dx: (6,) Cartesian delta [x, y, z, rx, ry, rz] in meters/radians
        """
        J = self.get_jacobian(q)
        dx = J @ dq
        return dx
    
    def cartesian_to_joint(self, dx: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Convert Cartesian-space delta to joint-space delta (inverse Jacobian).
        
        Args:
            dx: (6,) Cartesian delta [x, y, z, rx, ry, rz]
            q: (6,) current joint angles
            
        Returns:
            dq: (6,) joint angle deltas
        """
        J = self.get_jacobian(q)
        # Use pseudoinverse for robustness near singularities
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ dx
        return dq
    
    def joint_delta_to_serl_action(self, dq: np.ndarray, q: np.ndarray, 
                                    action_scale: np.ndarray = None) -> np.ndarray:
        """Convert VLA joint delta to SERL env action space.
        
        The SERL env action is in [-1, 1] and gets scaled by ACTION_SCALE:
          pos_delta = action[:3] * action_scale[0]  (default 0.01 → 10mm)
          rot_delta = action[3:6] * action_scale[1] / 4  (MRP, default 0.05)
        
        Args:
            dq: (6,) joint angle deltas from VLA/RLT
            q: (6,) current joint angles
            action_scale: [pos_scale, rot_scale] from SERL env config
            
        Returns:
            serl_action: (6,) in approximately [-1, 1] for SERL env
        """
        if action_scale is None:
            action_scale = np.array([0.01, 0.05])  # SERL default
        
        # Joint delta → Cartesian delta
        dx = self.joint_to_cartesian(dq, q)
        
        # Convert to SERL action space
        # Position: serl_action[:3] = dx[:3] / action_scale[0]
        serl_action = np.zeros(6, dtype=np.float32)
        serl_action[:3] = dx[:3] / action_scale[0]
        
        # Rotation: SERL uses MRP (Modified Rodrigues Parameters)
        # dx[3:6] is rotation vector. MRP = rotvec / 4 (small angle approx)
        # serl_action[3:6] = mrp / (action_scale[1] / 4) = rotvec / action_scale[1]
        serl_action[3:] = dx[3:] / action_scale[1]
        
        # Clip to [-1, 1] for safety
        serl_action = np.clip(serl_action, -1.0, 1.0)
        
        return serl_action
