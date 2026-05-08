"""Unit tests for rotation utilities."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "serl_robot_infra")

from ur_env.utils.rotations import (
    euler_2_quat,
    quat_2_euler,
    rotvec_2_quat,
    quat_2_rotvec,
    pose_2_quat,
    pose_2_rotvec,
    quat_2_mrp,
    mrp_2_quat,
)


class TestRotations:
    def test_euler_quat_roundtrip(self):
        euler = np.array([0.1, 0.2, 0.3])
        quat = euler_2_quat(euler)
        result = quat_2_euler(quat)
        assert np.allclose(euler, result, atol=1e-10)

    def test_euler_quat_zero(self):
        euler = np.array([0.0, 0.0, 0.0])
        quat = euler_2_quat(euler)
        # Identity quaternion: [0, 0, 0, 1]
        assert np.allclose(quat, [0, 0, 0, 1], atol=1e-10)

    def test_rotvec_quat_roundtrip(self):
        rotvec = np.array([0.5, -0.3, 0.1])
        quat = rotvec_2_quat(rotvec)
        result = quat_2_rotvec(quat)
        assert np.allclose(rotvec, result, atol=1e-10)

    def test_pose_2_quat_shape(self):
        pose6 = [0.3, -0.1, 0.08, 2.2, -2.24, 0.0]
        pose7 = pose_2_quat(pose6)
        assert pose7.shape == (7,)
        assert np.allclose(pose7[:3], [0.3, -0.1, 0.08])

    def test_pose_2_rotvec_roundtrip(self):
        pose6 = [0.3, -0.1, 0.08, 2.2, -2.24, 0.1]
        pose7 = pose_2_quat(pose6)
        pose6_back = pose_2_rotvec(pose7)
        assert np.allclose(pose6, pose6_back, atol=1e-6)

    def test_mrp_quat_roundtrip(self):
        from scipy.spatial.transform import Rotation as R
        euler = np.array([0.2, -0.1, 0.15])
        quat = R.from_euler("xyz", euler).as_quat()
        mrp = quat_2_mrp(quat)
        quat_back = mrp_2_quat(mrp)
        # Quaternions may differ by sign
        assert np.allclose(quat, quat_back, atol=1e-10) or \
               np.allclose(quat, -quat_back, atol=1e-10)

    def test_quat_norm(self):
        euler = np.array([1.0, 0.5, -0.3])
        quat = euler_2_quat(euler)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-10
