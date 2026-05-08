#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Train on LOCAL GPU (RTX 5070 Ti 16GB)
# TESTED & WORKING: 2.1 it/s, ~3h 53min for 30k steps
#
# Config: pi0_fast_ur5e_peg_insertion_lora
#   • π0-FAST model with LoRA rank=4
#   • Batch size: 1 × grad_accumulation=8 → effective batch=8
#   • 3 images (overhead + wrist x2)
#   • Uses 96% of 16GB VRAM
#
# IMPORTANT: Kill any zombie python processes before running!
#   nvidia-smi  (check for old python processes)
#   kill -9 <PID>  (kill them)
#
# GRADIENT ACCUMULATION:
#   Enabled by default (8 steps). To override:
#   ./train_local.sh <config> --grad-accumulation-steps=4
# ═══════════════════════════════════════════════════════════════════

# Pre-allocate GPU memory pool (avoids fragmentation)
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# Use 95% of GPU memory (default is 75% — not enough!)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Disable autotuning to save ~700MB during compilation
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo "════════════════════════════════════════════════════════"
echo "  Training on LOCAL GPU (RTX 5070 Ti 16GB)"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=true"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
echo "  XLA_FLAGS=--xla_gpu_autotune_level=0"
echo "  Gradient Accumulation: 8 steps (effective batch=8)"
echo ""
echo "  ⚠️  Make sure no other python/GPU processes are running!"
echo "     Check: nvidia-smi"
echo "════════════════════════════════════════════════════════"
echo ""

# Pass all arguments to train.py
uv run scripts/train.py "$@"
