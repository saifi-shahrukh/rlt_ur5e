"""
RLT Replay Buffer with chunked actions.

Designed for the RLT training loop where:
  - State = z_rl (token_dim,) + proprio (proprio_dim,)
  - Actions = chunks of C consecutive steps (C × action_dim)
  - Reference actions = VLA's predicted chunk ã (C × action_dim)
  - Stride-2 subsampling (paper Appendix B: ~25 samples/second from 50Hz)

Supports RLPD symmetric sampling: the learner samples 50% from this online
buffer and 50% from a separate demo buffer. Both buffers have identical
structure.
"""
from __future__ import annotations

from collections import deque
import random
from typing import Optional

import numpy as np


class RLTBuffer:
    """Replay buffer storing chunked transitions for RLT training.

    Each stored transition contains:
        - z_rl:         (token_dim,)          RL token at chunk start
        - proprio:      (proprio_dim,)        proprioception at chunk start
        - action_chunk: (chunk_size, act_dim) actual executed actions
        - ref_chunk:    (chunk_size, act_dim) VLA reference actions ã
        - reward:       scalar               episode reward (sparse)
        - z_rl_next:    (token_dim,)          RL token at chunk end
        - proprio_next: (proprio_dim,)        proprioception at chunk end
        - done:         bool                  episode termination flag
    """

    def __init__(
        self,
        capacity: int = 200_000,
        token_dim: int = 512,
        proprio_dim: int = 19,
        action_dim: int = 6,
        chunk_size: int = 10,
        stride: int = 2,
    ):
        """
        Args:
            capacity: maximum number of transitions to store
            token_dim: dimension of z_rl vector
            proprio_dim: dimension of proprioceptive state
            action_dim: single-step action dimension
            chunk_size: C — number of steps per action chunk
            stride: subsampling stride for chunking episodes
        """
        self.capacity = capacity
        self.token_dim = token_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.stride = stride

        self._buf: deque = deque(maxlen=capacity)

    def add_transition(
        self,
        z_rl: np.ndarray,
        proprio: np.ndarray,
        action_chunk: np.ndarray,
        ref_chunk: np.ndarray,
        reward: float,
        z_rl_next: np.ndarray,
        proprio_next: np.ndarray,
        done: float,
    ):
        """Add a single chunked transition directly."""
        self._buf.append({
            "z_rl": z_rl.astype(np.float32),
            "proprio": proprio.astype(np.float32),
            "action_chunk": action_chunk.astype(np.float32),
            "ref_chunk": ref_chunk.astype(np.float32),
            "reward": np.float32(reward),
            "z_rl_next": z_rl_next.astype(np.float32),
            "proprio_next": proprio_next.astype(np.float32),
            "done": np.float32(done),
        })

    def add_episode(
        self,
        z_rl_list: list[np.ndarray],
        proprio_list: list[np.ndarray],
        action_list: list[np.ndarray],
        ref_list: list[np.ndarray],
        reward: float,
        done: bool = True,
    ) -> int:
        """Add an episode by chunking with stride-2 subsampling.

        Takes per-step lists and creates chunked transitions at every
        `stride` steps. Only the last chunk in an episode gets the reward.

        Args:
            z_rl_list:    per-step RL tokens [T × (token_dim,)]
            proprio_list: per-step proprio [T × (proprio_dim,)]
            action_list:  per-step actions [T × (action_dim,)]
            ref_list:     per-step VLA reference actions [T × (action_dim,)]
            reward:       episode reward (binary: 1.0 success, 0.0 failure)
            done:         whether episode terminated

        Returns:
            Number of transitions added
        """
        T = len(action_list)
        C = self.chunk_size
        added = 0

        for t in range(0, T - C + 1, self.stride):
            # Create action and reference chunks
            a_chunk = np.stack(action_list[t:t + C])  # (C, action_dim)
            r_chunk = np.stack(ref_list[t:t + C])     # (C, action_dim)

            # Determine if this is the last chunk
            is_last = (t + C >= T - self.stride + 1)
            ep_reward = float(reward) if is_last else 0.0
            ep_done = float(done) if is_last else 0.0

            # Next state (at end of chunk)
            t_next = min(t + C, len(z_rl_list) - 1)

            self._buf.append({
                "z_rl": z_rl_list[t].astype(np.float32),
                "proprio": proprio_list[t].astype(np.float32),
                "action_chunk": a_chunk.astype(np.float32),
                "ref_chunk": r_chunk.astype(np.float32),
                "reward": np.float32(ep_reward),
                "z_rl_next": z_rl_list[t_next].astype(np.float32),
                "proprio_next": proprio_list[t_next].astype(np.float32),
                "done": np.float32(ep_done),
            })
            added += 1

        return added

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Returns:
            Dictionary with batched arrays:
                z_rl:         (B, token_dim)
                proprio:      (B, proprio_dim)
                action_chunk: (B, C, action_dim)
                ref_chunk:    (B, C, action_dim)
                reward:       (B,)
                z_rl_next:    (B, token_dim)
                proprio_next: (B, proprio_dim)
                done:         (B,)
        """
        n = min(batch_size, len(self._buf))
        batch = random.sample(list(self._buf), n)
        return {
            k: np.array([t[k] for t in batch])
            for k in batch[0].keys()
        }

    def sample_rlpd(
        self,
        batch_size: int,
        demo_buffer: Optional["RLTBuffer"] = None,
        demo_ratio: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """RLPD-style symmetric sampling: 50% demo + 50% online.

        Args:
            batch_size: total batch size
            demo_buffer: separate demo buffer (same structure)
            demo_ratio: fraction of batch from demo buffer

        Returns:
            Merged batch dictionary
        """
        if demo_buffer is None or len(demo_buffer) == 0:
            return self.sample(batch_size)

        n_demo = int(batch_size * demo_ratio)
        n_online = batch_size - n_demo

        b_demo = demo_buffer.sample(n_demo)
        b_online = self.sample(n_online)

        return {
            k: np.concatenate([b_demo[k], b_online[k]], axis=0)
            for k in b_online.keys()
        }

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_transitions: int = 500) -> bool:
        """Check if buffer has enough data to start training."""
        return len(self._buf) >= min_transitions

    def stats(self) -> dict:
        """Get buffer statistics."""
        if len(self._buf) == 0:
            return {"size": 0, "capacity": self.capacity}

        rewards = [t["reward"] for t in self._buf]
        return {
            "size": len(self._buf),
            "capacity": self.capacity,
            "fill_ratio": len(self._buf) / self.capacity,
            "total_reward": sum(rewards),
            "num_successes": sum(1 for r in rewards if r > 0),
        }
