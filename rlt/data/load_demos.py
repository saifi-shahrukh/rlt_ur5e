"""Load HIL-SERL demonstration data into RLTBuffer for RLPD training.

SERL demo format (from record_demos.py):
  List of transitions, each with:
    - observations: {"overview": (1,128,128,3), "state": (1,19), "wrist_1": (1,128,128,3)}
    - actions: (6,) Cartesian deltas [-1, 1]
    - next_observations: same structure
    - rewards: float (0 or 1)
    - masks: float (1.0)
    - dones: bool

We convert these into chunked transitions for the RLTBuffer:
  - z_rl: zeros (no VLA embeddings during demo collection)
  - proprio: state[0] (19D)
  - action_chunk: (chunk_size, 6) — padded/repeated if needed
  - ref_chunk: zeros (no VLA reference during demos)
  - reward: from demo data
  - done: from demo data
"""
from __future__ import annotations

import pickle
import glob
from pathlib import Path
from typing import Optional

import numpy as np

from rlt.agents.rlt_buffer import RLTBuffer


def load_serl_demos(
    demo_dir: str,
    buffer: RLTBuffer,
    chunk_size: int = 10,
    token_dim: int = 512,
    proprio_dim: int = 19,
    action_dim: int = 6,
    verbose: bool = True,
) -> int:
    """Load all PKL demo files from demo_dir into the RLTBuffer.

    Args:
        demo_dir: Directory containing *_transitions_*.pkl files
        buffer: RLTBuffer to fill
        chunk_size: Number of steps per action chunk
        token_dim: RL token dimension (zeros for demos)
        proprio_dim: Proprioceptive state dimension
        action_dim: Action dimension
        verbose: Print loading info

    Returns:
        Total number of transitions added to buffer
    """
    demo_files = sorted(glob.glob(str(Path(demo_dir) / "*.pkl")))
    if not demo_files:
        print(f"[DEMO] No PKL files found in {demo_dir}")
        return 0

    total_added = 0
    total_transitions = 0

    for fpath in demo_files:
        with open(fpath, "rb") as f:
            transitions = pickle.load(f)

        if not isinstance(transitions, list) or len(transitions) == 0:
            continue

        # Split into episodes (cut at done=True)
        episodes = []
        current_ep = []
        for t in transitions:
            current_ep.append(t)
            if t.get("dones", False):
                episodes.append(current_ep)
                current_ep = []
        if current_ep:  # Remaining transitions (incomplete episode)
            episodes.append(current_ep)

        for ep in episodes:
            added = _load_episode_to_buffer(
                ep, buffer, chunk_size, token_dim, proprio_dim, action_dim
            )
            total_added += added

        total_transitions += len(transitions)

        if verbose:
            print(f"[DEMO] Loaded {fpath}: {len(transitions)} trans, "
                  f"{len(episodes)} episodes")

    if verbose:
        print(f"[DEMO] Total: {total_transitions} transitions → "
              f"{total_added} chunks in buffer")

    return total_added


def _load_episode_to_buffer(
    episode: list,
    buffer: RLTBuffer,
    chunk_size: int,
    token_dim: int,
    proprio_dim: int,
    action_dim: int,
) -> int:
    """Convert a single episode into chunked transitions."""
    T = len(episode)
    if T < 1:
        return 0

    # Extract per-step data
    proprios = []
    actions = []
    rewards = []
    dones = []

    for t in episode:
        # State: (1, 19) → (19,)
        state = t["observations"]["state"]
        if hasattr(state, 'shape'):
            if state.ndim > 1:
                state = state[0]
            proprios.append(state[:proprio_dim].astype(np.float32))
        else:
            proprios.append(np.zeros(proprio_dim, dtype=np.float32))

        # Action: (6,) or (7,) — take first action_dim
        act = np.array(t["actions"], dtype=np.float32)
        actions.append(act[:action_dim])

        # Reward
        rewards.append(float(t.get("rewards", 0)))
        dones.append(bool(t.get("dones", False)))

    # Episode reward (last transition or sum)
    ep_reward = float(rewards[-1]) if rewards[-1] > 0 else float(any(r > 0 for r in rewards))
    ep_done = True  # demos are complete episodes

    # Create zero RL tokens (no VLA during demos)
    z_rl_zeros = np.zeros(token_dim, dtype=np.float32)

    # Create chunked transitions
    added = 0
    stride = max(1, chunk_size // 2)  # 50% overlap

    for start in range(0, T, stride):
        end = start + chunk_size

        # Pad if not enough steps left
        if end > T:
            # Pad with last action repeated
            chunk_actions = actions[start:T]
            pad_len = end - T
            chunk_actions += [actions[-1]] * pad_len
            chunk_actions = np.stack(chunk_actions)  # (chunk_size, action_dim)

            next_idx = T - 1
            is_last = True
        else:
            chunk_actions = np.stack(actions[start:end])  # (chunk_size, action_dim)
            next_idx = min(end, T - 1)
            is_last = (end >= T)

        # Reference chunk: zeros (no VLA during demo collection)
        ref_chunk = np.zeros_like(chunk_actions)

        # Assign reward only to last chunk of episode
        chunk_reward = float(ep_reward) if is_last else 0.0
        chunk_done = 1.0 if is_last else 0.0

        buffer.add_transition(
            z_rl=z_rl_zeros,
            proprio=proprios[start],
            action_chunk=chunk_actions,
            ref_chunk=ref_chunk,
            reward=chunk_reward,
            z_rl_next=z_rl_zeros,
            proprio_next=proprios[next_idx],
            done=chunk_done,
        )
        added += 1

        if is_last:
            break

    return added
