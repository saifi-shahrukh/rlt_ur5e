#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train RL Token encoder-decoder from pre-extracted embeddings
# Uses PyTorch (available in openpi venv) — much faster on GPU
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=rl_token
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/rl_token_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/rl_token_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
RLT="/data/beegfs/home/saifi/rlt_ur5e"
OPENPI="${RLT}/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
OUTPUT_DIR="${RLT}/checkpoints/rl_token"

# Which embeddings to train on (override with env vars)
VLA_CONFIG="${VLA_CONFIG:-pi0_fast_ur5e_peg_insertion_lora}"
VLA_STEP="${VLA_STEP:-4999}"
EMBEDDINGS_FILE="${OUTPUT_DIR}/embeddings_${VLA_CONFIG}_step${VLA_STEP}.pt"
SAVE_PATH="${OUTPUT_DIR}/${VLA_CONFIG}_rl_token.pt"

# Training hyperparameters
TOKEN_DIM="${TOKEN_DIM:-512}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV}"
export PYTHONPATH="${RLT}:${PYTHONPATH:-}"

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${RLT}"
mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Train RL Token Encoder-Decoder"
echo "  Embeddings: ${EMBEDDINGS_FILE}"
echo "  Save path:  ${SAVE_PATH}"
echo "  Token dim:  ${TOKEN_DIM}"
echo "  Steps:      ${STEPS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Start:      $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Verify embeddings exist
if [[ ! -f "${EMBEDDINGS_FILE}" ]]; then
    echo "  ERROR: Embeddings file not found: ${EMBEDDINGS_FILE}"
    echo "  Available embedding files:"
    ls -la ${OUTPUT_DIR}/embeddings_*.pt 2>/dev/null || echo "    (none)"
    echo ""
    echo "  Run extract_embeddings first:"
    echo "    VLA_CONFIG=${VLA_CONFIG} VLA_STEP=${VLA_STEP} sbatch hpc/slurm/extract_embeddings.sh"
    exit 1
fi
echo "  ✓ Embeddings found: $(du -h ${EMBEDDINGS_FILE} | cut -f1)"
echo ""

# ─── Train ───────────────────────────────────────────────────────────────────
${VENV}/bin/python3.11 -m rlt.training.train_rl_token \
    --cache "${EMBEDDINGS_FILE}" \
    --save_path "${SAVE_PATH}" \
    --token_dim "${TOKEN_DIM}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --device cuda

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ RL Token training complete! $(date)"
echo "  Model saved: ${SAVE_PATH}"
echo "  Size: $(du -h ${SAVE_PATH} | cut -f1)"
echo "═══════════════════════════════════════════════════════════════"
