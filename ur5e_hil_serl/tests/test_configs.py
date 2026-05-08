"""Unit tests for experiment configurations."""
import sys
import unittest.mock as mock

import numpy as np
import pytest

# Mock all hardware + ML modules
HW_MOCKS = {
    "pyrealsense2": mock.MagicMock(),
    "rtde_control": mock.MagicMock(),
    "rtde_receive": mock.MagicMock(),
    "pyspacemouse": mock.MagicMock(),
    "pylibfreenect2": mock.MagicMock(),
    "pynput": mock.MagicMock(),
    "pynput.keyboard": mock.MagicMock(),
    "cv2": mock.MagicMock(),
    "jax": mock.MagicMock(),
    "jax.numpy": mock.MagicMock(),
    "jax.random": mock.MagicMock(),
    "serl_launcher": mock.MagicMock(),
    "serl_launcher.wrappers": mock.MagicMock(),
    "serl_launcher.wrappers.serl_obs_wrappers": mock.MagicMock(),
    "serl_launcher.wrappers.chunking": mock.MagicMock(),
    "serl_launcher.networks": mock.MagicMock(),
    "serl_launcher.networks.reward_classifier": mock.MagicMock(),
}

sys.path.insert(0, "serl_robot_infra")
sys.path.insert(0, "examples")


class TestConfigMappings:
    def test_all_tasks_registered(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            from experiments.mappings import CONFIG_MAPPING
            expected_tasks = ["peg_insertion", "pcb_insertion", "bin_relocation", "cable_routing"]
            for task in expected_tasks:
                assert task in CONFIG_MAPPING, f"Task '{task}' not in CONFIG_MAPPING"

    def test_all_configs_instantiable(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            from experiments.mappings import CONFIG_MAPPING
            for name, cfg_cls in CONFIG_MAPPING.items():
                cfg = cfg_cls()
                assert cfg is not None, f"Failed to instantiate config for '{name}'"


class TestPegInsertionConfig:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            from experiments.peg_insertion.config import TrainConfig, EnvConfig
            self.train_cfg = TrainConfig()
            self.env_cfg = EnvConfig()

    def test_setup_mode(self):
        assert self.train_cfg.setup_mode == "single-arm-fixed-gripper"

    def test_image_keys(self):
        assert "wrist_1" in self.train_cfg.image_keys
        assert "overview" in self.train_cfg.image_keys

    def test_proprio_keys(self):
        expected = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
        for key in expected:
            assert key in self.train_cfg.proprio_keys

    def test_env_config_gripper(self):
        assert self.env_cfg.GRIPPER_RELEASE_ON_RESET is False

    def test_env_config_action_scale(self):
        assert self.env_cfg.ACTION_SCALE[0] == 0.005  # 1cm per step
        assert self.env_cfg.ACTION_SCALE[2] == 1.0  # full gripper range

    def test_safety_box_contains_target(self):
        target = self.env_cfg.TARGET_POSE[:3]
        low = self.env_cfg.ABS_POSE_LIMIT_LOW[:3]
        high = self.env_cfg.ABS_POSE_LIMIT_HIGH[:3]
        assert np.all(target >= low), f"Target {target} below safety low {low}"
        assert np.all(target <= high), f"Target {target} above safety high {high}"


class TestPCBInsertionConfig:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            from experiments.pcb_insertion.config import TrainConfig, EnvConfig
            self.train_cfg = TrainConfig()
            self.env_cfg = EnvConfig()

    def test_setup_mode(self):
        assert self.train_cfg.setup_mode == "single-arm-fixed-gripper"

    def test_finer_action_scale(self):
        # PCB should have finer action scale than peg
        assert self.env_cfg.ACTION_SCALE[0] == 0.005

    def test_gripper_stays_closed(self):
        assert self.env_cfg.GRIPPER_RELEASE_ON_RESET is False


class TestBinRelocationConfig:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            from experiments.bin_relocation.config import TrainConfig, EnvConfig
            self.train_cfg = TrainConfig()
            self.env_cfg = EnvConfig()

    def test_setup_mode(self):
        assert self.train_cfg.setup_mode == "single-arm-learned-gripper"

    def test_gripper_releases(self):
        assert self.env_cfg.GRIPPER_RELEASE_ON_RESET is True

    def test_larger_action_scale(self):
        # Pick-place needs bigger steps
        assert self.env_cfg.ACTION_SCALE[0] > 0.01


class TestCableRoutingConfig:
    def setup_method(self):
        with mock.patch.dict(sys.modules, HW_MOCKS):
            from experiments.cable_routing.config import TrainConfig, EnvConfig
            self.train_cfg = TrainConfig()
            self.env_cfg = EnvConfig()

    def test_setup_mode(self):
        assert self.train_cfg.setup_mode == "single-arm-fixed-gripper"

    def test_larger_workspace(self):
        # Cable routing needs bigger workspace
        box_size = self.env_cfg.ABS_POSE_LIMIT_HIGH[:3] - self.env_cfg.ABS_POSE_LIMIT_LOW[:3]
        assert box_size[0] >= 0.4  # at least 40cm X range

    def test_larger_action_scale(self):
        assert self.env_cfg.ACTION_SCALE[0] == 0.02  # 2cm per step

    def test_longer_episodes(self):
        assert self.env_cfg.MAX_EPISODE_LENGTH == 150
