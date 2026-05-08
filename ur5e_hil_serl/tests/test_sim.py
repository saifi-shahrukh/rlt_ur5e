"""Unit tests for ur5e_sim MuJoCo environment."""
import sys
import numpy as np
import pytest

sys.path.insert(0, "ur5e_sim")


class TestUR5eSimImport:
    def test_import_module(self):
        import ur5e_sim
        assert hasattr(ur5e_sim, '__name__')

    def test_import_env(self):
        from ur5e_sim.envs.ur5e_pick_gym_env import UR5ePickCubeGymEnv
        assert UR5ePickCubeGymEnv is not None

    def test_import_controller(self):
        from ur5e_sim.controllers import opspace
        assert callable(opspace)

    def test_import_mujoco_gym_env(self):
        from ur5e_sim.mujoco_gym_env import MujocoGymEnv, GymRenderingSpec
        assert MujocoGymEnv is not None
        assert GymRenderingSpec is not None


class TestUR5eSimEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ur5e_sim.envs.ur5e_pick_gym_env import UR5ePickCubeGymEnv
        self.env = UR5ePickCubeGymEnv(render_mode="rgb_array", image_obs=False)
        yield
        self.env.close()

    def test_reset(self):
        obs, info = self.env.reset()
        assert "state" in obs
        assert "ur5e/tcp_pos" in obs["state"]
        assert "ur5e/tcp_vel" in obs["state"]
        assert "ur5e/gripper_pos" in obs["state"]
        assert "block_pos" in obs["state"]

    def test_obs_shapes(self):
        obs, _ = self.env.reset()
        assert obs["state"]["ur5e/tcp_pos"].shape == (3,)
        assert obs["state"]["ur5e/tcp_vel"].shape == (3,)
        assert obs["state"]["ur5e/gripper_pos"].shape == (1,)
        assert obs["state"]["block_pos"].shape == (3,)

    def test_action_space(self):
        assert self.env.action_space.shape == (4,)
        assert np.all(self.env.action_space.low == -1.0)
        assert np.all(self.env.action_space.high == 1.0)

    def test_step(self):
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        assert "state" in obs
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)

    def test_multiple_steps(self):
        self.env.reset()
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(action)
            if done:
                break
        assert obs["state"]["ur5e/tcp_pos"].shape == (3,)

    def test_reward_range(self):
        self.env.reset()
        rewards = []
        for _ in range(20):
            _, r, _, _, _ = self.env.step(self.env.action_space.sample())
            rewards.append(r)
        # Reward should be bounded [0, 1] by design
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_render_rgb_array(self):
        self.env.reset()
        frames = self.env.render()
        assert isinstance(frames, list)
        assert len(frames) == 2  # front + wrist
        for frame in frames:
            assert frame.shape == (128, 128, 3)
            assert frame.dtype == np.uint8


class TestUR5eSimVision:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ur5e_sim.envs.ur5e_pick_gym_env import UR5ePickCubeGymEnv
        self.env = UR5ePickCubeGymEnv(render_mode="rgb_array", image_obs=True)
        yield
        self.env.close()

    def test_vision_obs(self):
        obs, _ = self.env.reset()
        assert "images" in obs
        assert "front" in obs["images"]
        assert "wrist" in obs["images"]
        assert obs["images"]["front"].shape == (128, 128, 3)
        assert obs["images"]["wrist"].shape == (128, 128, 3)

    def test_vision_state_keys(self):
        obs, _ = self.env.reset()
        assert "ur5e/tcp_pos" in obs["state"]
        assert "ur5e/tcp_vel" in obs["state"]
        assert "ur5e/gripper_pos" in obs["state"]
        # In vision mode, block_pos should NOT be in state
        assert "block_pos" not in obs["state"]


class TestUR5eSimGymRegistry:
    def test_state_env_registered(self):
        import gymnasium as gym
        import ur5e_sim  # triggers registration
        env = gym.make("UR5ePickCube-v0")
        obs, _ = env.reset()
        assert "state" in obs
        env.close()

    def test_vision_env_registered(self):
        import gymnasium as gym
        import ur5e_sim
        env = gym.make("UR5ePickCubeVision-v0")
        obs, _ = env.reset()
        assert "images" in obs
        env.close()
