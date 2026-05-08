#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Start VLA Server (π0-FAST fine-tuned on peg insertion)
# ═══════════════════════════════════════════════════════════════════════════
# Run this in Terminal 1. Wait for "Server ready" before starting inference.
# ═══════════════════════════════════════════════════════════════════════════
set -e

VLA_DIR="/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e"
CONFIG="pi0_fast_ur5e_peg_insertion_lora"
CHECKPOINT_DIR="checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999"
PORT=8000

# Allow override
[[ -n "$1" ]] && CHECKPOINT_DIR="$1"
[[ -n "$2" ]] && PORT="$2"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  VLA Server — ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Port: ws://localhost:${PORT}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

cd "${VLA_DIR}"
exec .venv/bin/python scripts/serve_policy.py \
  --port ${PORT} \
  policy:checkpoint \
  --policy.config ${CONFIG} \
  --policy.dir ${CHECKPOINT_DIR}
