#!/usr/bin/env python3
"""
Extract VLM prefix embeddings from demo data using the OpenPI venv.

This script MUST be run in the openpi virtual environment:
  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
  source .venv/bin/activate
  cd ~/ur5e_hande_workspace/rlt_ur5e

  python rlt/training/extract_embeddings.py \
    --config_name pi0_fast_ur5e_peg_insertion_lora \
    --checkpoint_dir openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999 \
    --output checkpoints/rl_token/embeddings_peg_insertion.pt \
    --n_samples 200

Then train RL Token (in hilserl venv):
  source ur5e_hil_serl/.venv/bin/activate
  python -m rlt.training.train_rl_token \
    --cache checkpoints/rl_token/embeddings_peg_insertion.pt \
    --save_path checkpoints/rl_token/peg_insertion_v1.pt \
    --steps 5000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM embeddings (run in openpi venv)"
    )
    parser.add_argument("--config_name", type=str,
                        default="pi0_fast_ur5e_peg_insertion_lora")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999")
    parser.add_argument("--output", type=str,
                        default="checkpoints/rl_token/embeddings_peg_insertion.pt")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to extract (with random variation)")
    parser.add_argument("--demo_root", type=str, default=None,
                        help="LeRobot dataset path (optional, uses random if not set)")
    args = parser.parse_args()

    # Verify openpi is available
    try:
        # Add openpi src to path
        openpi_src = Path("openpi_ur5e/openpi-ur5e/src")
        if openpi_src.exists():
            sys.path.insert(0, str(openpi_src))

        from openpi.training import config as _config
        from openpi.policies import policy_config
        from openpi.models import model as _model
        import jax
        import jax.numpy as jnp
        import torch
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nThis script must be run in the openpi venv:")
        print("  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e")
        print("  source .venv/bin/activate")
        print("  cd ~/ur5e_hande_workspace/rlt_ur5e")
        print("  python rlt/training/extract_embeddings.py")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load policy ──────────────────────────────────────────────────────
    print(f"[Extract] Config: {args.config_name}")
    print(f"[Extract] Checkpoint: {args.checkpoint_dir}")
    cfg = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    model = policy._model
    print(f"[Extract] Model loaded: {type(model).__name__}")

    # ── Try to load real demo frames ─────────────────────────────────────
    demo_frames = []
    if args.demo_root:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(
                repo_id="saifi/ur5e-peg-insertion-dual",
                root=str(Path(args.demo_root).parent),
            )
            print(f"[Extract] Loaded dataset: {len(dataset)} frames")
            for idx in range(min(len(dataset), args.n_samples * 3)):
                sample = dataset[idx]
                # Convert to expected format
                frame = {}
                for key in sample:
                    val = sample[key]
                    if hasattr(val, 'numpy'):
                        val = val.numpy()
                    if "wrist_cam" in key:
                        if val.ndim == 3 and val.shape[0] == 3:
                            val = val.transpose(1, 2, 0)
                        val = (val * 255).astype(np.uint8) if val.max() <= 1.0 else val
                        frame["observation/wrist_image_left"] = val
                        frame["observation/wrist_image_right"] = val
                    elif "overview_cam" in key:
                        if val.ndim == 3 and val.shape[0] == 3:
                            val = val.transpose(1, 2, 0)
                        val = (val * 255).astype(np.uint8) if val.max() <= 1.0 else val
                        frame["observation/exterior_image_1_left"] = val
                    elif "state" in key:
                        frame["observation/joint_position"] = val.astype(np.float32)
                if len(frame) >= 3:  # Need at least images + state
                    frame["prompt"] = "Pick up the peg and insert it into the hole."
                    demo_frames.append(frame)
            print(f"[Extract] Prepared {len(demo_frames)} frames from dataset")
        except Exception as e:
            print(f"[Extract] Could not load dataset: {e}")
            print("[Extract] Falling back to random observations")

    # ── Generate observations (real or random) ───────────────────────────
    print(f"[Extract] Extracting {args.n_samples} embeddings...")
    embeddings = []
    start_time = time.time()

    for i in range(args.n_samples):
        # Use real demo frame if available, else random
        if demo_frames and i < len(demo_frames):
            obs = demo_frames[i]
        else:
            # Random observation with slight variations
            # (not ideal but gives structurally valid embeddings for testing)
            obs = {
                "observation/joint_position": np.random.randn(7).astype(np.float32) * 0.5,
                "observation/exterior_image_1_left": np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8),
                "observation/wrist_image_left": np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8),
                "observation/wrist_image_right": np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8),
                "prompt": "Pick up the peg and insert it into the hole.",
            }

        # Transform inputs (same as policy.infer)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        # Get prefix embeddings
        emb, input_mask, ar_mask = model.embed_inputs(observation)
        # emb shape: (1, N_prefix, 2048)

        # Convert to numpy and store
        emb_np = np.array(emb[0], dtype=np.float32)  # (N_prefix, 2048)
        embeddings.append(torch.tensor(emb_np))

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{args.n_samples}] shape={emb_np.shape} "
                  f"({rate:.1f} samples/s)")

    # ── Save ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n[Extract] Extracted {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"[Extract] Shape: {embeddings[0].shape}")
    print(f"[Extract] Saving to: {output_path}")

    torch.save({
        "embeddings": embeddings,
        "config": {
            "config_name": args.config_name,
            "checkpoint_dir": args.checkpoint_dir,
            "n_samples": len(embeddings),
            "embed_dim": int(embeddings[0].shape[-1]),
            "n_tokens": int(embeddings[0].shape[0]),
            "used_real_demos": len(demo_frames) > 0,
        },
    }, output_path)

    print(f"\n[Extract] DONE! ✓")
    print(f"")
    print(f"Next step (in hilserl venv):")
    print(f"  source ur5e_hil_serl/.venv/bin/activate")
    print(f"  python -m rlt.training.train_rl_token \\")
    print(f"    --cache {output_path} \\")
    print(f"    --save_path checkpoints/rl_token/peg_insertion_v1.pt \\")
    print(f"    --steps 5000")


if __name__ == "__main__":
    main()
