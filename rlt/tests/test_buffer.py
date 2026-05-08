"""
Unit tests for the RLT replay buffer.

Run with:
    cd ~/ur5e_hande_workspace/rlt_ur5e
    python -m pytest rlt/tests/test_buffer.py -v

Or standalone:
    python rlt/tests/test_buffer.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from rlt.agents.rlt_buffer import RLTBuffer


def test_buffer_creation():
    """Test buffer instantiation."""
    buf = RLTBuffer(capacity=1000, token_dim=512, proprio_dim=19,
                    action_dim=6, chunk_size=10)
    assert len(buf) == 0
    assert buf.capacity == 1000
    print(f"  Buffer created: capacity={buf.capacity}")


def test_add_single_transition():
    """Test adding a single transition."""
    buf = RLTBuffer(capacity=100, token_dim=32, proprio_dim=7,
                    action_dim=6, chunk_size=10)

    buf.add_transition(
        z_rl=np.random.randn(32).astype(np.float32),
        proprio=np.random.randn(7).astype(np.float32),
        action_chunk=np.random.randn(10, 6).astype(np.float32),
        ref_chunk=np.random.randn(10, 6).astype(np.float32),
        reward=1.0,
        z_rl_next=np.random.randn(32).astype(np.float32),
        proprio_next=np.random.randn(7).astype(np.float32),
        done=1.0,
    )
    assert len(buf) == 1
    print(f"  Single transition added: len={len(buf)}")


def test_add_episode():
    """Test adding a full episode with stride-2 chunking."""
    buf = RLTBuffer(capacity=1000, token_dim=32, proprio_dim=7,
                    action_dim=6, chunk_size=10, stride=2)

    T = 50  # Episode length
    z_rl_list = [np.random.randn(32).astype(np.float32) for _ in range(T)]
    proprio_list = [np.random.randn(7).astype(np.float32) for _ in range(T)]
    action_list = [np.random.randn(6).astype(np.float32) for _ in range(T)]
    ref_list = [np.random.randn(6).astype(np.float32) for _ in range(T)]

    added = buf.add_episode(
        z_rl_list=z_rl_list,
        proprio_list=proprio_list,
        action_list=action_list,
        ref_list=ref_list,
        reward=1.0,
        done=True,
    )

    # With T=50, C=10, stride=2: range(0, 50-10+1, 2) = range(0, 41, 2) = 21 chunks
    expected = len(range(0, T - 10 + 1, 2))
    assert added == expected, f"Expected {expected} transitions, got {added}"
    assert len(buf) == added
    print(f"  Episode (T={T}): {added} transitions added (stride=2, C=10)")


def test_add_episode_reward_assignment():
    """Test that only the last chunk gets the reward."""
    buf = RLTBuffer(capacity=1000, token_dim=16, proprio_dim=7,
                    action_dim=6, chunk_size=5, stride=1)

    T = 20
    z_rl_list = [np.zeros(16, dtype=np.float32) for _ in range(T)]
    proprio_list = [np.zeros(7, dtype=np.float32) for _ in range(T)]
    action_list = [np.zeros(6, dtype=np.float32) for _ in range(T)]
    ref_list = [np.zeros(6, dtype=np.float32) for _ in range(T)]

    buf.add_episode(
        z_rl_list=z_rl_list,
        proprio_list=proprio_list,
        action_list=action_list,
        ref_list=ref_list,
        reward=10.0,
        done=True,
    )

    # Check that only last transition has reward
    rewards = [t["reward"] for t in buf._buf]
    assert rewards[-1] == 10.0, f"Last reward should be 10.0, got {rewards[-1]}"
    assert sum(rewards[:-1]) == 0.0, "Non-last transitions should have 0 reward"
    print(f"  Reward assignment: only last chunk gets reward ✓")


def test_sample_shapes():
    """Test that sampled batch has correct shapes."""
    buf = RLTBuffer(capacity=1000, token_dim=64, proprio_dim=19,
                    action_dim=6, chunk_size=10, stride=2)

    # Add some data
    for _ in range(5):
        T = 40
        buf.add_episode(
            z_rl_list=[np.random.randn(64).astype(np.float32) for _ in range(T)],
            proprio_list=[np.random.randn(19).astype(np.float32) for _ in range(T)],
            action_list=[np.random.randn(6).astype(np.float32) for _ in range(T)],
            ref_list=[np.random.randn(6).astype(np.float32) for _ in range(T)],
            reward=1.0,
        )

    batch = buf.sample(8)

    assert batch["z_rl"].shape == (8, 64)
    assert batch["proprio"].shape == (8, 19)
    assert batch["action_chunk"].shape == (8, 10, 6)
    assert batch["ref_chunk"].shape == (8, 10, 6)
    assert batch["reward"].shape == (8,)
    assert batch["z_rl_next"].shape == (8, 64)
    assert batch["proprio_next"].shape == (8, 19)
    assert batch["done"].shape == (8,)

    print(f"  Sample shapes correct:")
    print(f"    z_rl: {batch['z_rl'].shape}")
    print(f"    proprio: {batch['proprio'].shape}")
    print(f"    action_chunk: {batch['action_chunk'].shape}")
    print(f"    ref_chunk: {batch['ref_chunk'].shape}")


def test_sample_rlpd():
    """Test RLPD-style 50/50 sampling from demo + online buffers."""
    online_buf = RLTBuffer(capacity=1000, token_dim=32, proprio_dim=7,
                           action_dim=6, chunk_size=5, stride=1)
    demo_buf = RLTBuffer(capacity=1000, token_dim=32, proprio_dim=7,
                         action_dim=6, chunk_size=5, stride=1)

    # Fill online buffer with zeros
    for _ in range(10):
        T = 20
        online_buf.add_episode(
            z_rl_list=[np.zeros(32, dtype=np.float32) for _ in range(T)],
            proprio_list=[np.zeros(7, dtype=np.float32) for _ in range(T)],
            action_list=[np.zeros(6, dtype=np.float32) for _ in range(T)],
            ref_list=[np.zeros(6, dtype=np.float32) for _ in range(T)],
            reward=0.0,
        )

    # Fill demo buffer with ones
    for _ in range(10):
        T = 20
        demo_buf.add_episode(
            z_rl_list=[np.ones(32, dtype=np.float32) for _ in range(T)],
            proprio_list=[np.ones(7, dtype=np.float32) for _ in range(T)],
            action_list=[np.ones(6, dtype=np.float32) for _ in range(T)],
            ref_list=[np.ones(6, dtype=np.float32) for _ in range(T)],
            reward=1.0,
        )

    batch = online_buf.sample_rlpd(16, demo_buffer=demo_buf, demo_ratio=0.5)

    assert batch["z_rl"].shape[0] == 16
    # Should have mix of zeros (online) and ones (demo)
    has_zeros = np.any(batch["z_rl"].sum(axis=1) == 0)
    has_ones = np.any(batch["z_rl"].sum(axis=1) > 0)
    assert has_zeros and has_ones, "RLPD sampling should mix both buffers"
    print(f"  RLPD sampling: mixed batch of 16 (50% demo + 50% online) ✓")


def test_capacity_overflow():
    """Test that buffer properly evicts old data when full."""
    buf = RLTBuffer(capacity=10, token_dim=8, proprio_dim=4,
                    action_dim=3, chunk_size=3, stride=1)

    # Add more than capacity
    for i in range(20):
        buf.add_transition(
            z_rl=np.full(8, i, dtype=np.float32),
            proprio=np.zeros(4, dtype=np.float32),
            action_chunk=np.zeros((3, 3), dtype=np.float32),
            ref_chunk=np.zeros((3, 3), dtype=np.float32),
            reward=0.0,
            z_rl_next=np.zeros(8, dtype=np.float32),
            proprio_next=np.zeros(4, dtype=np.float32),
            done=0.0,
        )

    assert len(buf) == 10, f"Should be capped at 10, got {len(buf)}"
    # Oldest entries should be gone (FIFO eviction)
    oldest_z = buf._buf[0]["z_rl"][0]
    assert oldest_z >= 10, f"Oldest entry should be >=10, got {oldest_z}"
    print(f"  Capacity overflow: correctly capped at {len(buf)} ✓")


def test_is_ready():
    """Test buffer readiness check."""
    buf = RLTBuffer(capacity=1000, token_dim=16, proprio_dim=7,
                    action_dim=6, chunk_size=5, stride=1)

    assert not buf.is_ready(min_transitions=10)

    # Add enough data
    for i in range(15):
        buf.add_transition(
            z_rl=np.zeros(16, dtype=np.float32),
            proprio=np.zeros(7, dtype=np.float32),
            action_chunk=np.zeros((5, 6), dtype=np.float32),
            ref_chunk=np.zeros((5, 6), dtype=np.float32),
            reward=0.0,
            z_rl_next=np.zeros(16, dtype=np.float32),
            proprio_next=np.zeros(7, dtype=np.float32),
            done=0.0,
        )

    assert buf.is_ready(min_transitions=10)
    print(f"  is_ready: correctly reports readiness ✓")


def test_stats():
    """Test buffer statistics."""
    buf = RLTBuffer(capacity=100, token_dim=16, proprio_dim=7,
                    action_dim=6, chunk_size=5, stride=1)

    stats = buf.stats()
    assert stats["size"] == 0

    # Add some episodes with mixed rewards
    for reward in [1.0, 0.0, 1.0, 0.0, 0.0]:
        T = 10
        buf.add_episode(
            z_rl_list=[np.zeros(16, dtype=np.float32) for _ in range(T)],
            proprio_list=[np.zeros(7, dtype=np.float32) for _ in range(T)],
            action_list=[np.zeros(6, dtype=np.float32) for _ in range(T)],
            ref_list=[np.zeros(6, dtype=np.float32) for _ in range(T)],
            reward=reward,
        )

    stats = buf.stats()
    assert stats["size"] > 0
    assert stats["num_successes"] == 2  # Two episodes with reward=1.0
    print(f"  Stats: size={stats['size']}, successes={stats['num_successes']} ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def run_all_tests():
    """Run all buffer tests."""
    tests = [
        ("Buffer creation", test_buffer_creation),
        ("Add single transition", test_add_single_transition),
        ("Add episode (stride-2)", test_add_episode),
        ("Reward assignment", test_add_episode_reward_assignment),
        ("Sample shapes", test_sample_shapes),
        ("RLPD sampling", test_sample_rlpd),
        ("Capacity overflow", test_capacity_overflow),
        ("is_ready()", test_is_ready),
        ("stats()", test_stats),
    ]

    print("\n" + "=" * 60)
    print("  RLT Buffer — Unit Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            print(f"[TEST] {name}")
            test_fn()
            passed += 1
            print(f"  ✅ PASSED\n")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ❌ FAILED: {e}\n")

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  • {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
