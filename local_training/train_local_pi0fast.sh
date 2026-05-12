#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Local Training: pi0-FAST on RTX 5070 Ti (16GB)
# Uses rank=4 LoRA, batch=1 + grad_accum=8
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

OPENPI="${HOME}/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e"
cd "${OPENPI}"
source .venv/bin/activate

# Memory optimization for 16GB GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
export JAX_TRACEBACK_FILTERING=off

# Config
CONFIG="pi0_fast_ur5e_peg_insertion_lora"  # rank=4 version (fits 16GB)
EXP_NAME="${1:-local_experiment}"
STEPS="${2:-5000}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Local Training: pi0-FAST (rank=4)"
echo "  Exp: ${EXP_NAME}"
echo "  Steps: ${STEPS}"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --batch-size=1 \
    --grad-accumulation-steps=8 \
    --num-workers=2 \
    --num-train-steps=${STEPS} \
    --save-interval=1000

echo ""
echo "  ✓ Done! Checkpoint: checkpoints/${CONFIG}/${EXP_NAME}/"
