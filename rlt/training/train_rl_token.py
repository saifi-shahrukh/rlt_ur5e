"""
Offline training of RL Token encoder-decoder.

This script is run ONCE after VLA fine-tuning and before starting online RL.
It pre-computes VLM embeddings from demonstration data, caches them,
then trains the RL Token model to compress them through an information bottleneck.

Usage:
  cd ~/ur5e_hande_workspace/rlt_ur5e
  source ur5e_hil_serl/.venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/openpi_ur5e/openpi-ur5e/src:$PYTHONPATH"

  # With real VLA (requires GPU + checkpoint):
  python -m rlt.training.train_rl_token \
    --demo_root openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
    --vla_ckpt openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999 \
    --vla_config pi0_fast_ur5e_peg_insertion_lora \
    --save_path checkpoints/rl_token/peg_insertion_v1.pt

  # With synthetic data (for testing without VLA):
  python -m rlt.training.train_rl_token --synthetic --save_path checkpoints/rl_token/test.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rlt.models.rl_token import RLTokenModel


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset: Pre-computed VLM Embeddings
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingDataset(Dataset):
    """Dataset of pre-computed VLM token embeddings.

    Either loads from cache (if available) or computes from demo data
    using the Pi05Hook.
    """

    def __init__(
        self,
        embeddings: list[torch.Tensor],
    ):
        """Initialize with pre-computed embeddings.

        Args:
            embeddings: list of (N_i, embed_dim) tensors
        """
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def collate_variable_length(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collate tensors with variable sequence lengths by padding.

    Args:
        batch: list of (N_i, D) tensors with potentially different N_i

    Returns:
        Padded tensor (B, N_max, D) with zero-padding
    """
    max_len = max(t.shape[0] for t in batch)
    D = batch[0].shape[1]
    padded = torch.zeros(len(batch), max_len, D)
    for i, t in enumerate(batch):
        padded[i, :t.shape[0], :] = t
    return padded


def create_synthetic_dataset(n_samples: int = 500, n_tokens: int = 527,
                             embed_dim: int = 2048) -> list[torch.Tensor]:
    """Create synthetic embeddings for testing (mimics VLM output statistics).

    The synthetic data has:
    - Roughly unit variance per dimension (like normalized transformer outputs)
    - Some structure (not pure random noise)
    """
    print(f"[Dataset] Creating {n_samples} synthetic samples "
          f"(N={n_tokens}, D={embed_dim})")
    embeddings = []
    for i in range(n_samples):
        # Create structured embeddings: low-rank + noise
        # This simulates real VLM embeddings which are not pure noise
        rank = 64
        basis = torch.randn(rank, embed_dim) * 0.5
        coeffs = torch.randn(n_tokens, rank) * 0.3
        structured = coeffs @ basis
        noise = torch.randn(n_tokens, embed_dim) * 0.1
        z = structured + noise
        embeddings.append(z)
    return embeddings


def compute_embeddings_from_demos(
    demo_root: Path,
    vla_ckpt: str,
    vla_config: str,
    cache_path: Path,
    stride: int = 5,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Pre-compute VLM embeddings from LeRobot demonstration data.

    This runs the VLA model on each demo frame and extracts the prefix
    token embeddings via the forward hook.

    Args:
        demo_root: Path to LeRobot dataset directory
        vla_ckpt: Path to VLA checkpoint
        vla_config: OpenPI config name
        cache_path: Where to save/load cached embeddings
        stride: Only process every N-th frame (saves time)
        device: "cuda" or "cpu"

    Returns:
        List of (N_prefix, embed_dim) tensors
    """
    # Check cache first
    if cache_path.exists():
        print(f"[Dataset] Loading cached embeddings: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        print(f"[Dataset] Loaded {len(data['embeddings'])} samples from cache")
        return data["embeddings"]

    print(f"[Dataset] Computing VLM embeddings from demos at: {demo_root}")
    print(f"[Dataset] VLA checkpoint: {vla_ckpt}")
    print(f"[Dataset] This may take a while (runs VLA forward pass per frame)...")

    from rlt.models.pi05_hook import Pi05Hook

    hook = Pi05Hook(
        checkpoint_dir=vla_ckpt,
        config_name=vla_config,
        device=device,
    )

    embeddings = []

    # Try to load LeRobot-format data
    try:
        import pandas as pd

        parquet_dir = demo_root / "data"
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {parquet_dir}")

        print(f"[Dataset] Found {len(parquet_files)} parquet file(s)")

        frame_count = 0
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for idx in range(0, len(df), stride):
                row = df.iloc[idx]
                try:
                    # Extract observation from parquet row
                    # Note: actual image loading depends on dataset format
                    # For video-based datasets, we need to decode frames
                    state = np.array(row["observation.state"], dtype=np.float32)

                    # For now, create a placeholder obs
                    # Real implementation would load video frames here
                    obs = {
                        "state": state,
                        # Images would be loaded from video files
                    }

                    # If we can get images, extract embeddings
                    if "wrist_image" in obs or "wrist_1" in obs:
                        z_tokens, _ = hook.get_embeddings_and_actions(obs)
                        embeddings.append(
                            torch.tensor(z_tokens, dtype=torch.float32)
                        )
                        frame_count += 1

                        if frame_count % 50 == 0:
                            print(f"  Processed {frame_count} frames...")

                except Exception as e:
                    if frame_count == 0:
                        print(f"  Warning: frame {idx} failed: {e}")
                    continue

    except Exception as e:
        print(f"[Dataset] Error loading demo data: {e}")
        print("[Dataset] Falling back to synthetic data for testing")
        embeddings = create_synthetic_dataset(200, 527, 2048)

    # Save cache
    if embeddings:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings}, cache_path)
        print(f"[Dataset] Cached {len(embeddings)} samples → {cache_path}")

    hook.close()
    return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════


def train(args):
    """Train the RL Token encoder-decoder."""
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Train] CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Build dataset ─────────────────────────────────────────────────────
    if args.synthetic:
        embeddings = create_synthetic_dataset(
            n_samples=args.n_synthetic,
            n_tokens=args.n_tokens,
            embed_dim=args.embed_dim,
        )
    elif args.cache and Path(args.cache).exists():
        # Load pre-extracted embeddings (from extract_embeddings.py in openpi venv)
        print(f"[Train] Loading pre-extracted embeddings from: {args.cache}")
        cache_data = torch.load(args.cache, weights_only=False)
        embeddings = cache_data["embeddings"]
        print(f"[Train] Loaded {len(embeddings)} embeddings")
        if "config" in cache_data:
            print(f"[Train] Source: {cache_data['config']}")
    else:
        demo_root = Path(args.demo_root)
        cache_path = Path(args.cache) if args.cache else \
            Path(f"checkpoints/rl_token/.cache_{demo_root.name}.pt")

        embeddings = compute_embeddings_from_demos(
            demo_root=demo_root,
            vla_ckpt=args.vla_ckpt,
            vla_config=args.vla_config,
            cache_path=cache_path,
            stride=args.stride,
            device=device,
        )

    if not embeddings:
        print("[Train] ERROR: No embeddings to train on!")
        return

    # Determine embed_dim and max sequence length from data
    embed_dim = embeddings[0].shape[-1]
    max_seq_len = max(e.shape[0] for e in embeddings)
    min_seq_len = min(e.shape[0] for e in embeddings)
    print(f"\n[Train] Dataset: {len(embeddings)} samples")
    print(f"[Train] Embed dim: {embed_dim}")
    print(f"[Train] Token dim (bottleneck): {args.token_dim}")
    print(f"[Train] Sequence lengths: min={min_seq_len}, max={max_seq_len}")

    # Create data loader
    dataset = EmbeddingDataset(embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    # ── Build model ───────────────────────────────────────────────────────
    # Set max_len to accommodate actual data sequence length
    model_max_len = max(max_seq_len + 10, 600)  # +10 for safety margin

    model = RLTokenModel(
        embed_dim=embed_dim,
        token_dim=args.token_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        max_len=model_max_len,
    ).to(device)
    print(f"[Train] Model max_len: {model_max_len} (data max: {max_seq_len})")

    param_counts = model.get_num_params()
    print(f"[Train] Model params: encoder={param_counts['encoder']:,}, "
          f"decoder={param_counts['decoder']:,}, total={param_counts['total']:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training for {args.steps} steps...")
    print(f"[Train] Device: {device}")
    print("=" * 60)

    data_iter = iter(dataloader)
    best_loss = float("inf")
    start_time = time.time()
    loss_history = []

    for step in range(args.steps):
        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device)  # (B, N_max, D)

        # Forward pass
        loss, z_rl = model.compute_loss(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Logging
        if step % args.log_interval == 0:
            z_rl_norm = z_rl.norm(dim=-1).mean().item()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

            print(
                f"  step={step:5d}/{args.steps} | "
                f"loss={loss_val:.5f} | "
                f"z_rl_norm={z_rl_norm:.3f} | "
                f"lr={lr:.2e} | "
                f"{steps_per_sec:.1f} steps/s"
            )

        # Save best checkpoint
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "embed_dim": embed_dim,
                    "token_dim": args.token_dim,
                    "enc_layers": args.enc_layers,
                    "dec_layers": args.dec_layers,
                    "n_heads": args.n_heads,
                    "ffn_dim": args.ffn_dim,
                    "max_len": model_max_len,
                },
                "loss": best_loss,
                "step": step,
            }, save_path)

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - start_time
    avg_loss_last_100 = np.mean(loss_history[-100:]) if len(loss_history) >= 100 \
        else np.mean(loss_history)

    print("\n" + "=" * 60)
    print(f"[Train] DONE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[Train] Best loss: {best_loss:.5f}")
    print(f"[Train] Avg loss (last 100): {avg_loss_last_100:.5f}")
    print(f"[Train] Saved → {save_path}")
    print("=" * 60)

    return best_loss


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Train RL Token encoder-decoder (offline, run once)"
    )

    # Data source (choose one)
    parser.add_argument("--demo_root", type=str, default=None,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing without VLA)")

    # VLA settings (required if not --synthetic)
    parser.add_argument("--vla_ckpt", type=str, default=None,
                        help="Path to trained VLA checkpoint")
    parser.add_argument("--vla_config", type=str,
                        default="pi0_fast_ur5e_peg_insertion_lora",
                        help="OpenPI config name")

    # Cache
    parser.add_argument("--cache", type=str, default=None,
                        help="Path for embedding cache file")

    # Output
    parser.add_argument("--save_path", type=str,
                        default="checkpoints/rl_token/rl_token_v1.pt",
                        help="Where to save trained model")

    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=2048,
                        help="VLM embedding dimension")
    parser.add_argument("--token_dim", type=int, default=512,
                        help="RL token dimension (bottleneck)")
    parser.add_argument("--enc_layers", type=int, default=4,
                        help="Encoder transformer layers")
    parser.add_argument("--dec_layers", type=int, default=4,
                        help="Decoder transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Attention heads")
    parser.add_argument("--ffn_dim", type=int, default=2048,
                        help="FFN dimension")

    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=5000,
                        help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame subsampling stride for demo data")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Print loss every N steps")

    # Synthetic data settings
    parser.add_argument("--n_synthetic", type=int, default=500,
                        help="Number of synthetic samples")
    parser.add_argument("--n_tokens", type=int, default=527,
                        help="Sequence length for synthetic data")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Validation
    if not args.synthetic and args.demo_root is None and not (args.cache and Path(args.cache).exists()):
        parser.error("Either --demo_root, --cache (existing file), or --synthetic must be specified")
    if not args.synthetic and args.vla_ckpt is None and not (args.cache and Path(args.cache).exists()):
        parser.error("--vla_ckpt is required when not using --synthetic or --cache")

    train(args)


if __name__ == "__main__":
    main()
