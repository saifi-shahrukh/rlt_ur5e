"""Integration tests for the full HIL-SERL pipeline.

Tests the complete flow: config → env → agent → training step
without requiring real hardware.

NOTE: These tests import real JAX/TF modules and must not be run
in the same pytest session as test_configs.py/test_env.py (which mock them).
Run separately: python -m pytest tests/test_pipeline.py -v
"""
import sys
import os
import unittest.mock as mock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pytest

# Skip entire module if JAX has been mocked by another test file in this session
try:
    import jax
    if isinstance(jax, mock.MagicMock) or not hasattr(jax, 'devices'):
        pytest.skip("JAX is mocked — run test_pipeline.py separately", allow_module_level=True)
    # Also check that JAX is actually functional (not just imported)
    _ = jax.devices()
except Exception:
    pytest.skip("JAX not available — run test_pipeline.py separately", allow_module_level=True)

pytestmark = pytest.mark.integration

sys.path.insert(0, "serl_robot_infra")
sys.path.insert(0, "examples")


class TestPegInsertionPipeline:
    """End-to-end pipeline test for peg insertion (fake env + GPU)."""

    def test_env_creation_fake(self):
        from experiments.peg_insertion.config import TrainConfig
        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        assert env.observation_space is not None
        assert env.action_space.shape == (6,)

    def test_env_obs_structure(self):
        from experiments.peg_insertion.config import TrainConfig
        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        obs = env.observation_space.sample()
        assert "state" in obs
        assert "wrist_1" in obs
        assert "overview" in obs
        # State should be flattened: tcp_pose(6) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper_pose(1) = 19
        assert obs["state"].shape[-1] == 19

    def test_agent_creation(self):
        import jax
        from experiments.peg_insertion.config import TrainConfig
        from serl_launcher.utils.launcher import make_sac_pixel_agent

        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        agent = make_sac_pixel_agent(
            seed=42,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        assert agent is not None

    def test_agent_sample_actions(self):
        import jax
        from experiments.peg_insertion.config import TrainConfig
        from serl_launcher.utils.launcher import make_sac_pixel_agent

        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        agent = make_sac_pixel_agent(
            seed=42,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        rng = jax.random.PRNGKey(0)
        obs = env.observation_space.sample()
        actions = agent.sample_actions(
            observations=jax.device_put(obs), seed=rng, argmax=False
        )
        actions_np = np.array(jax.device_get(actions))
        assert actions_np.shape == (6,)
        assert np.all(np.abs(actions_np) <= 1.0)

    def test_replay_buffer_insert_and_sample(self):
        import jax
        from experiments.peg_insertion.config import TrainConfig
        from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=500,
            image_keys=config.image_keys,
        )

        for _ in range(200):
            transition = dict(
                observations=env.observation_space.sample(),
                actions=env.action_space.sample(),
                next_observations=env.observation_space.sample(),
                rewards=float(np.random.rand() > 0.9),
                masks=1.0,
                dones=False,
            )
            replay_buffer.insert(transition)

        assert len(replay_buffer) >= 200

    def test_training_step(self):
        import jax
        import jax.numpy as jnp
        from experiments.peg_insertion.config import TrainConfig
        from serl_launcher.utils.launcher import make_sac_pixel_agent
        from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)

        devices = jax.local_devices()
        sharding = jax.sharding.PositionalSharding(devices)

        agent = make_sac_pixel_agent(
            seed=42,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=500,
            image_keys=config.image_keys,
        )
        for _ in range(256):
            transition = dict(
                observations=env.observation_space.sample(),
                actions=env.action_space.sample(),
                next_observations=env.observation_space.sample(),
                rewards=float(np.random.rand() > 0.9),
                masks=1.0,
                dones=False,
            )
            replay_buffer.insert(transition)

        replay_iterator = replay_buffer.get_iterator(
            sample_args={"batch_size": 128, "pack_obs_and_next_obs": True},
            device=sharding.replicate(),
        )

        batch = next(replay_iterator)
        train_networks = frozenset({"critic", "actor", "temperature"})
        agent, update_info = agent.update(batch, networks_to_update=train_networks)

        assert "critic" in update_info
        assert "actor" in update_info
        assert "temperature" in update_info


class TestBCPipeline:
    """Test behavior cloning pipeline."""

    def test_bc_agent_creation(self):
        import jax
        from experiments.peg_insertion.config import TrainConfig
        from serl_launcher.utils.launcher import make_bc_agent

        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        bc_agent = make_bc_agent(
            seed=42,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )
        assert bc_agent is not None

    def test_bc_training_step(self):
        import jax
        import jax.numpy as jnp
        from experiments.peg_insertion.config import TrainConfig
        from serl_launcher.utils.launcher import make_bc_agent
        from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

        config = TrainConfig()
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)

        devices = jax.local_devices()
        sharding = jax.sharding.PositionalSharding(devices)

        bc_agent = make_bc_agent(
            seed=42,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )
        bc_agent = jax.device_put(jax.tree.map(jnp.array, bc_agent), sharding.replicate())

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=500,
            image_keys=config.image_keys,
        )
        for _ in range(256):
            transition = dict(
                observations=env.observation_space.sample(),
                actions=env.action_space.sample(),
                next_observations=env.observation_space.sample(),
                rewards=0.0,
                masks=1.0,
                dones=False,
            )
            replay_buffer.insert(transition)

        replay_iterator = replay_buffer.get_iterator(
            sample_args={"batch_size": 64, "pack_obs_and_next_obs": False},
            device=sharding.replicate(),
        )

        batch = next(replay_iterator)
        bc_agent, update_info = bc_agent.update(batch)
        assert "actor_loss" in update_info or "mse" in update_info


class TestAllTaskConfigs:
    """Verify all registered tasks can create environments."""

    def test_all_tasks_create_env(self):
        from experiments.mappings import CONFIG_MAPPING
        for name, cfg_cls in CONFIG_MAPPING.items():
            config = cfg_cls()
            env = config.get_environment(fake_env=True, save_video=False, classifier=False)
            obs = env.observation_space.sample()
            action = env.action_space.sample()
            assert obs is not None, f"Task '{name}' failed to sample obs"
            assert action is not None, f"Task '{name}' failed to sample action"
