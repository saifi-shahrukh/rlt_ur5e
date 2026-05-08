"""Rotation utilities for UR5e HIL-SERL.

Provides the same interface as hil-serl's franka_env/utils/rotations.py
plus UR-specific helpers.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_2_quat(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (xyz convention) to quaternion [x, y, z, w]."""
    return R.from_euler("xyz", euler).as_quat()


def quat_2_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to euler angles (xyz convention)."""
    return R.from_quat(quat).as_euler("xyz")


def rotvec_2_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to quaternion [x, y, z, w]."""
    return R.from_rotvec(rotvec).as_quat()


def quat_2_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to rotation vector."""
    return R.from_quat(quat).as_rotvec()


def pose_2_quat(pose_6d: list) -> np.ndarray:
    """Convert UR TCP pose [x, y, z, rx, ry, rz] to [x, y, z, qx, qy, qz, qw].

    UR uses rotation vector (axis-angle) representation for orientation.
    """
    pos = np.array(pose_6d[:3], dtype=np.float64)
    quat = R.from_rotvec(pose_6d[3:]).as_quat()
    return np.concatenate([pos, quat])


def pose_2_rotvec(pose_7d: np.ndarray) -> list:
    """Convert [x, y, z, qx, qy, qz, qw] to UR format [x, y, z, rx, ry, rz]."""
    pos = pose_7d[:3].tolist()
    rotvec = R.from_quat(pose_7d[3:]).as_rotvec().tolist()
    return pos + rotvec


def quat_2_mrp(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Modified Rodrigues Parameters."""
    return R.from_quat(quat).as_mrp()


def mrp_2_quat(mrp: np.ndarray) -> np.ndarray:
    """Convert Modified Rodrigues Parameters to quaternion."""
    return R.from_mrp(mrp).as_quat()
