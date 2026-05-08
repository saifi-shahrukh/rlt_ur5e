#!/usr/bin/env python3
"""
Extract VLM prefix embeddings from REAL demo observations.

This script MUST be run in the openpi virtual environment:
  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
  source .venv/bin/activate
  cd ~/ur5e_hande_workspace/rlt_ur5e

  python rlt/training/extract_embeddings_real.py \
    --output checkpoints/rl_token/embeddings_peg_insertion_9demos.pt

Then train RL Token (in hilserl venv):
  source ur5e_hil_serl/.venv/bin/activate
  python -m rlt.training.train_rl_token \
    --cache checkpoints/rl_token/embeddings_peg_insertion_9demos.pt \
    --save_path checkpoints/rl_token/peg_insertion_9demos_v1.pt \
    --steps 5000 --token_dim 512
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def decode_video_frames(video_path, max_frames=None):
    """Decode all frames from an AV1/mp4 video using pyav."""
    import av
    frames = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if max_frames and i >= max_frames:
            break
        frames.append(frame.to_ndarray(format='rgb24'))
    container.close()
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM embeddings from REAL demo frames (run in openpi venv)"
    )
    parser.add_argument("--config_name", type=str,
                        default="pi0_fast_ur5e_peg_insertion_lora")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_9demos/29999")
    parser.add_argument("--dataset_dir", type=str,
                        default="openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual")
    parser.add_argument("--output", type=str,
                        default="checkpoints/rl_token/embeddings_peg_insertion_9demos.pt")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to extract (None = all frames)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride (1=every frame, 2=every other)")
    args = parser.parse_args()

    # Verify openpi is available
    try:
        openpi_src = Path("openpi_ur5e/openpi-ur5e/src")
        if openpi_src.exists():
            sys.path.insert(0, str(openpi_src))

        from openpi.training import config as _config
        from openpi.policies import policy_config
        from openpi.models import model as _model
        import jax
        import jax.numpy as jnp
        import torch
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nThis script must be run in the openpi venv.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    # ── Load policy ──────────────────────────────────────────────────────
    print(f"[Extract] Config: {args.config_name}")
    print(f"[Extract] Checkpoint: {args.checkpoint_dir}")
    print(f"[Extract] Dataset: {args.dataset_dir}")
    cfg = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    model = policy._model
    print(f"[Extract] Model loaded: {type(model).__name__}")

    # ── Load demo data ───────────────────────────────────────────────────
    # Read state data from parquet
    parquet_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(str(parquet_path))
    states = t.column('observation.state').to_pylist()
    episode_indices = t.column('episode_index').to_pylist()
    n_total_frames = len(states)
    n_episodes = len(set(episode_indices))
    print(f"[Extract] Loaded parquet: {n_total_frames} frames, {n_episodes} episodes")

    # Decode video frames
    wrist_video = dataset_dir / "videos" / "observation.images.wrist_cam" / "chunk-000" / "file-000.mp4"
    overview_video = dataset_dir / "videos" / "observation.images.overview_cam" / "chunk-000" / "file-000.mp4"

    print(f"[Extract] Decoding wrist video...")
    wrist_frames = decode_video_frames(wrist_video)
    print(f"[Extract] Decoded {len(wrist_frames)} wrist frames, shape={wrist_frames[0].shape}")

    print(f"[Extract] Decoding overview video...")
    overview_frames = decode_video_frames(overview_video)
    print(f"[Extract] Decoded {len(overview_frames)} overview frames, shape={overview_frames[0].shape}")

    assert len(wrist_frames) == len(overview_frames) == n_total_frames, \
        f"Frame count mismatch: wrist={len(wrist_frames)}, overview={len(overview_frames)}, parquet={n_total_frames}"

    # Apply stride
    indices = list(range(0, n_total_frames, args.stride))
    if args.max_samples and args.max_samples < len(indices):
        step = len(indices) // args.max_samples
        indices = indices[::step][:args.max_samples]

    n_extract = len(indices)
    print(f"\n[Extract] Will extract {n_extract} embeddings (stride={args.stride})")

    # ── Extract embeddings ───────────────────────────────────────────────
    embeddings = []
    start_time = time.time()

    for count, i in enumerate(indices):
        # Get real data for this frame
        wrist_img = wrist_frames[i]      # (480, 640, 3) uint8
        overview_img = overview_frames[i]  # (480, 640, 3) uint8
        state = np.array(states[i], dtype=np.float32)

        # Build observation dict matching openpi input format
        obs = {
            "observation/joint_position": state,
            "observation/exterior_image_1_left": overview_img,
            "observation/wrist_image_left": wrist_img,
            "observation/wrist_image_right": wrist_img,  # same cam for both
            "prompt": "Pick up the peg and insert it into the hole.",
        }

        # Transform and get embeddings
        inputs = policy._input_transform(obs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        emb, input_mask, ar_mask = model.embed_inputs(observation)
        # emb shape: (1, N_tokens, embed_dim) bfloat16

        # Convert to float32 numpy then torch
        emb_np = np.array(emb[0], dtype=np.float32)  # (N_tokens, embed_dim)
        embeddings.append(torch.tensor(emb_np))

        if (count + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (count + 1) / elapsed
            eta = (n_extract - count - 1) / rate
            print(f"  [{count+1}/{n_extract}] shape={emb_np.shape} "
                  f"norm={np.linalg.norm(emb_np, axis=-1).mean():.1f} "
                  f"({rate:.1f} samples/s, ETA: {eta:.0f}s)")

    # ── Save ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n[Extract] Extracted {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"[Extract] Shape: {embeddings[0].shape}")

    # Compute stats
    norms = [e.norm(dim=-1).mean().item() for e in embeddings]
    print(f"[Extract] Embedding norm: mean={np.mean(norms):.1f}, "
          f"std={np.std(norms):.1f}, min={np.min(norms):.1f}, max={np.max(norms):.1f}")

    print(f"[Extract] Saving to: {output_path}")
    torch.save({
        "embeddings": embeddings,
        "config": {
            "config_name": args.config_name,
            "checkpoint_dir": args.checkpoint_dir,
            "dataset_dir": str(dataset_dir),
            "n_samples": len(embeddings),
            "embed_dim": int(embeddings[0].shape[-1]),
            "n_tokens": int(embeddings[0].shape[0]),
            "used_real_demos": True,
            "stride": args.stride,
            "n_episodes": n_episodes,
        },
    }, output_path)

    print(f"\n[Extract] DONE! ✓")
    print(f"")
    print(f"Next step (in hilserl venv):")
    print(f"  source ur5e_hil_serl/.venv/bin/activate")
    print(f"  python -m rlt.training.train_rl_token \\")
    print(f"    --cache {output_path} \\")
    print(f"    --save_path checkpoints/rl_token/peg_insertion_9demos_v1.pt \\")
    print(f"    --steps 5000 --token_dim 512")


if __name__ == "__main__":
    main()
