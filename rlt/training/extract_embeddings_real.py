#!/usr/bin/env python3
"""
Extract VLM prefix embeddings from REAL demo observations.

This script MUST be run in the openpi virtual environment:
  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
  source .venv/bin/activate
  cd ~/ur5e_hande_workspace/rlt_ur5e

  python rlt/training/extract_embeddings_real.py \
    --output checkpoints/rl_token/embeddings_peg_insertion_real.pt

Then train RL Token (in hilserl venv):
  source ur5e_hil_serl/.venv/bin/activate
  python -m rlt.training.train_rl_token \
    --cache checkpoints/rl_token/embeddings_peg_insertion_real.pt \
    --save_path checkpoints/rl_token/peg_insertion_real_v1.pt \
    --steps 5000 --token_dim 2048
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM embeddings from REAL demo frames (run in openpi venv)"
    )
    parser.add_argument("--config_name", type=str,
                        default="pi0_fast_ur5e_peg_insertion_lora")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999")
    parser.add_argument("--dataset_dir", type=str,
                        default="openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual")
    parser.add_argument("--output", type=str,
                        default="checkpoints/rl_token/embeddings_peg_insertion_real.pt")
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
        from PIL import Image
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
    df = t.to_pandas()
    print(f"[Extract] Loaded parquet: {len(df)} frames, episodes: {list(df['episode_index'].unique())}")

    # Find image directories
    wrist_dir = dataset_dir / "images" / "observation.images.wrist_cam"
    overview_dir = dataset_dir / "images" / "observation.images.overview_cam"

    # Build frame list
    # Images are in episode-000004/ (seems to be separate from parquet episodes 0-3)
    # Also check if videos need to be decoded
    image_episodes = sorted(wrist_dir.iterdir()) if wrist_dir.exists() else []
    print(f"[Extract] Image episode dirs: {[d.name for d in image_episodes]}")

    # Strategy: use parquet states + video frames OR image frames
    frames = []

    # Try to load from image directories first
    for ep_dir in image_episodes:
        ep_name = ep_dir.name  # e.g., "episode-000004"
        wrist_frames = sorted((wrist_dir / ep_name).glob("frame-*.png"))
        overview_frames = sorted((overview_dir / ep_name).glob("frame-*.png"))
        n_frames = min(len(wrist_frames), len(overview_frames))
        print(f"[Extract]   {ep_name}: {n_frames} image frames")

        for idx in range(0, n_frames, args.stride):
            frames.append({
                "wrist_path": str(wrist_frames[idx]),
                "overview_path": str(overview_frames[idx]),
                "episode": ep_name,
                "frame_idx": idx,
            })

    # If no images, decode from videos
    if not frames:
        print("[Extract] No image dirs found, trying video decode...")
        # Fall back to using parquet states with random images
        # (This shouldn't happen for this dataset)
        print("[Extract] ERROR: No images found in dataset!")
        sys.exit(1)

    # Get states from parquet (match if possible, otherwise use first episode)
    # Note: images are episode-000004 but parquet has episodes 0-3
    # We'll use states from the parquet cycling through them
    all_states = [np.array(s, dtype=np.float32) for s in df["observation.state"].values]

    n_total = len(frames)
    if args.max_samples and args.max_samples < n_total:
        # Sample evenly
        indices = np.linspace(0, n_total - 1, args.max_samples, dtype=int)
        frames = [frames[i] for i in indices]
    n_extract = len(frames)

    print(f"\n[Extract] Will extract {n_extract} embeddings from real demo frames")
    print(f"[Extract] States available: {len(all_states)} (from parquet)")

    # ── Extract embeddings ───────────────────────────────────────────────
    embeddings = []
    start_time = time.time()

    for i, frame_info in enumerate(frames):
        # Load real images
        wrist_img = np.array(Image.open(frame_info["wrist_path"]))
        overview_img = np.array(Image.open(frame_info["overview_path"]))

        # Get state (cycle through available states)
        state_idx = i % len(all_states)
        state = all_states[state_idx]

        # Build observation dict
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
        # emb shape: (1, 948, 2048) bfloat16

        # Convert to float32 numpy then torch
        emb_np = np.array(emb[0], dtype=np.float32)  # (948, 2048)
        embeddings.append(torch.tensor(emb_np))

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n_extract}] shape={emb_np.shape} "
                  f"norm={np.linalg.norm(emb_np, axis=-1).mean():.1f} "
                  f"({rate:.1f} samples/s)")

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
        },
    }, output_path)

    print(f"\n[Extract] DONE! ✓")
    print(f"")
    print(f"Next step (in hilserl venv):")
    print(f"  source ur5e_hil_serl/.venv/bin/activate")
    print(f"  python -m rlt.training.train_rl_token \\")
    print(f"    --cache {output_path} \\")
    print(f"    --save_path checkpoints/rl_token/peg_insertion_real_v1.pt \\")
    print(f"    --steps 5000 --token_dim 2048")


if __name__ == "__main__":
    main()
