"""Unit tests for UR5eEnv (fake mode) and wrappers."""
import sys
import unittest.mock as mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Mock hardware modules before importing ur_env
HW_MOCKS = {
    "pyrealsense2": mock.MagicMock(),
    "rtde_control": mock.MagicMock(),
    "rtde_receive": mock.MagicMock(),
    "pyspacemouse": mock.MagicMock(),
    "pylibfreenect2": mock.MagicMock(),
    "pynput": mock.MagicMock(),
    "pynput.keyboard": mock.MagicMock(),
}

sys.path.insert(0, "serl_robot_infra")

with mock.patch.dict(sys.modules, HW_MOCKS):
    from ur_env.envs.ur5e_env import UR5eEnv, DefaultEnvConfig
    from ur_env.envs.relative_env import RelativeFrame, construct_homogeneous_matrix
    from ur_env.envs.wrappers import (
        Quat2EulerWrapper,
        GripperCloseEnv,
        MultiCameraBinaryRewardClassifierWrapper,
        GripperPenaltyWrapper,
    )
    from ur_env.spacemouse.spacemouse_expert import FakeSpaceMouseExpert


class TestDefaultEnvConfig:
    def test_config_attributes(self):
        config = DefaultEnvConfig()
        assert config.ROBOT_IP == "172.22.1.139"
        assert config.CONTROLLER_HZ == 100
        assert config.MAX_EPISODE_LENGTH == 100
        assert config.ACTION_SCALE.shape == (3,)
        assert config.ABS_POSE_LIMIT_HIGH.shape == (6,)
        assert config.ABS_POSE_LIMIT_LOW.shape == (6,)

    def test_config_cameras(self):
        config = DefaultEnvConfig()
        assert "wrist_1" in config.REALSENSE_CAMERAS
        assert "overview" in config.KINECT_CAMERAS


class TestUR5eEnvFake:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            self.config = DefaultEnvConfig()
            self.env = UR5eEnv(fake_env=True, config=self.config)

    def test_observation_space_structure(self):
        obs_space = self.env.observation_space
        assert "state" in obs_space.spaces
        assert "images" in obs_space.spaces

    def test_state_space_keys(self):
        state_space = self.env.observation_space["state"]
        expected_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "tcp_force", "tcp_torque"]
        for key in expected_keys:
            assert key in state_space.spaces, f"Missing state key: {key}"

    def test_state_space_shapes(self):
        state_space = self.env.observation_space["state"]
        assert state_space["tcp_pose"].shape == (7,)
        assert state_space["tcp_vel"].shape == (6,)
        assert state_space["gripper_pose"].shape == (1,)
        assert state_space["tcp_force"].shape == (3,)
        assert state_space["tcp_torque"].shape == (3,)

    def test_image_space_keys(self):
        img_space = self.env.observation_space["images"]
        assert "wrist_1" in img_space.spaces
        assert "overview" in img_space.spaces

    def test_image_space_shape(self):
        img_space = self.env.observation_space["images"]
        assert img_space["wrist_1"].shape == (128, 128, 3)
        assert img_space["overview"].shape == (128, 128, 3)

    def test_action_space(self):
        assert self.env.action_space.shape == (7,)
        assert np.all(self.env.action_space.low == -1.0)
        assert np.all(self.env.action_space.high == 1.0)

    def test_clip_safety_box(self):
        # Create a pose inside bounds
        pos = np.array([0.25, 0.0, 0.15])
        quat = R.from_euler("xyz", [3.0, 0.0, 0.0]).as_quat()
        pose = np.concatenate([pos, quat])
        clipped = self.env.clip_safety_box(pose.copy())
        assert clipped[:3][0] == 0.25  # inside, unchanged

        # Create a pose outside X bounds
        pose_out = np.concatenate([np.array([10.0, 0.0, 0.15]), quat])
        clipped_out = self.env.clip_safety_box(pose_out.copy())
        assert clipped_out[0] == self.config.ABS_POSE_LIMIT_HIGH[0]


class TestGripperCloseEnv:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            config = DefaultEnvConfig()
            base_env = UR5eEnv(fake_env=True, config=config)
            self.env = GripperCloseEnv(base_env)

    def test_action_space_reduced(self):
        assert self.env.action_space.shape == (6,)

    def test_action_wrapping(self):
        action_6d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        action_7d = self.env.action(action_6d)
        assert action_7d.shape == (7,)
        assert np.allclose(action_7d[:6], action_6d)
        assert action_7d[6] == -1.0  # gripper closed


class TestQuat2EulerWrapper:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            config = DefaultEnvConfig()
            base_env = UR5eEnv(fake_env=True, config=config)
            self.env = Quat2EulerWrapper(base_env)

    def test_tcp_pose_shape_converted(self):
        assert self.env.observation_space["state"]["tcp_pose"].shape == (6,)

    def test_observation_conversion(self):
        # Simulate an observation with quaternion
        euler_orig = np.array([0.1, 0.2, 0.3])
        quat = R.from_euler("xyz", euler_orig).as_quat()
        obs = {
            "state": {
                "tcp_pose": np.concatenate([np.array([0.3, 0.1, 0.08]), quat]),
                "tcp_vel": np.zeros(6),
                "gripper_pose": np.array([0.5]),
                "tcp_force": np.zeros(3),
                "tcp_torque": np.zeros(3),
            },
            "images": {},
        }
        converted = self.env.observation(obs)
        assert converted["state"]["tcp_pose"].shape == (6,)
        assert np.allclose(converted["state"]["tcp_pose"][:3], [0.3, 0.1, 0.08])
        assert np.allclose(converted["state"]["tcp_pose"][3:], euler_orig, atol=1e-6)


class TestRelativeFrame:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            config = DefaultEnvConfig()
            base_env = UR5eEnv(fake_env=True, config=config)
            self.env = RelativeFrame(base_env)

    def test_wraps_env(self):
        assert self.env.observation_space["state"]["tcp_pose"].shape == (7,)

    def test_transform_action_identity(self):
        # With identity rotation matrix, action should be unchanged
        self.env.rotation_matrix = np.eye(3)
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        transformed = self.env.transform_action(action.copy())
        assert np.allclose(transformed, action)

    def test_transform_action_rotation(self):
        # 90-degree rotation around Z
        self.env.rotation_matrix = R.from_euler("z", 90, degrees=True).as_matrix()
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        transformed = self.env.transform_action(action.copy())
        # X in body frame → Y in base frame (approx)
        assert abs(transformed[0]) < 1e-10  # x ≈ 0
        assert abs(transformed[1] - 1.0) < 1e-10  # y ≈ 1

    def test_construct_homogeneous_matrix(self):
        pose = np.array([0.3, -0.1, 0.08, 0.0, 0.0, 0.0, 1.0])
        T = construct_homogeneous_matrix(pose)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], [0.3, -0.1, 0.08])
        assert np.allclose(T[:3, :3], np.eye(3))  # identity rotation
        assert np.allclose(T[3, :], [0, 0, 0, 1])


class TestFakeSpaceMouse:
    def test_returns_zeros(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            sm = FakeSpaceMouseExpert()
            action, buttons = sm.get_action()
            assert action.shape == (6,)
            assert np.allclose(action, 0.0)
            assert buttons == (False, False)


class TestMultiCameraRewardWrapper:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            config = DefaultEnvConfig()
            base_env = UR5eEnv(fake_env=True, config=config)
            self.reward_fn = lambda obs: 1  # always success
            self.env = MultiCameraBinaryRewardClassifierWrapper(base_env, self.reward_fn)

    def test_wrapper_exists(self):
        assert self.env is not None
        assert self.env.reward_classifier_func is not None
