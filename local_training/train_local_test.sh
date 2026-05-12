#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Quick Test: 100 steps to verify config works
# Run this before committing to full HPC training
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

OPENPI="${HOME}/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e"
cd "${OPENPI}"
source .venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
export JAX_TRACEBACK_FILTERING=off

# Choose config (default: pi0-FAST which fits on 16GB)
CONFIG="${1:-pi0_fast_ur5e_peg_insertion_lora}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Quick Test: ${CONFIG}"
echo "  Steps: 100 (just to verify it works)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python scripts/train.py ${CONFIG} \
    --exp-name=test_run \
    --overwrite \
    --batch-size=1 \
    --grad-accumulation-steps=4 \
    --num-workers=2 \
    --num-train-steps=100 \
    --save-interval=50

echo ""
echo "  ✓ Test passed! Config ${CONFIG} works."
echo "  You can now submit to HPC with full settings."

# Clean up test checkpoint
rm -rf "checkpoints/${CONFIG}/test_run"
echo "  (cleaned up test checkpoint)"
