#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Extract VLM Embeddings from trained VLA checkpoint
# Runs in the openpi venv, outputs .pt file for RL Token training
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=extract_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/extract_emb_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/extract_emb_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
RLT="/data/beegfs/home/saifi/rlt_ur5e"
VENV="${OPENPI}/.venv"
OUTPUT_DIR="${RLT}/checkpoints/rl_token"

# Which model to extract from (override with environment variable)
VLA_CONFIG="${VLA_CONFIG:-pi0_fast_ur5e_peg_insertion_lora}"
VLA_STEP="${VLA_STEP:-4999}"
N_SAMPLES="${N_SAMPLES:-200}"

VLA_CHECKPOINT="${OPENPI}/checkpoints/${VLA_CONFIG}/peg_insertion_50demos/${VLA_STEP}"
OUTPUT_FILE="${OUTPUT_DIR}/embeddings_${VLA_CONFIG}_step${VLA_STEP}.pt"

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV}"
export PYTHONPATH="${OPENPI}/src:${RLT}:${PYTHONPATH:-}"

# HuggingFace
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export HF_TOKEN=$(cat /data/beegfs/home/saifi/.cache/huggingface/token 2>/dev/null || echo "")
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${RLT}"
mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Extract VLM Embeddings"
echo "  Config:     ${VLA_CONFIG}"
echo "  Checkpoint: ${VLA_CHECKPOINT}"
echo "  Output:     ${OUTPUT_FILE}"
echo "  Samples:    ${N_SAMPLES}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Start:      $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Verify checkpoint exists
if [[ ! -d "${VLA_CHECKPOINT}" ]]; then
    echo "  ERROR: Checkpoint not found: ${VLA_CHECKPOINT}"
    echo "  Available checkpoints:"
    ls -d ${OPENPI}/checkpoints/${VLA_CONFIG}/peg_insertion_50demos/[0-9]* 2>/dev/null || echo "    (none)"
    exit 1
fi
echo "  ✓ Checkpoint found"
echo ""

# ─── Extract ─────────────────────────────────────────────────────────────────
${VENV}/bin/python3.11 rlt/training/extract_embeddings.py \
    --config_name "${VLA_CONFIG}" \
    --checkpoint_dir "${VLA_CHECKPOINT}" \
    --output "${OUTPUT_FILE}" \
    --n_samples "${N_SAMPLES}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Embedding extraction complete! $(date)"
echo "  Output: ${OUTPUT_FILE}"
echo "  Size: $(du -h ${OUTPUT_FILE} | cut -f1)"
echo "═══════════════════════════════════════════════════════════════"
