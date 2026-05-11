#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0-FAST LoRA (50 demos peg insertion)
# Config: pi0_fast_ur5e_peg_insertion_lora | batch_size=1 | 30k steps
# Dataset: saifi/ur5e-peg-insertion-50demos-v2
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi0fast_50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi0fast_50_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi0fast_50_%j.err

# Don't use set -e: patchelf failures shouldn't kill the job
# set -e

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi0_fast_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_50demos_v2"

# ─── Environment (system commands work fine here - no LD_LIBRARY_PATH yet) ───
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
export CONDA_PREFIX="${VENV}"

# HuggingFace: token for gated models + offline mode (dataset is local)
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export HF_TOKEN=$(cat /data/beegfs/home/saifi/.cache/huggingface/token 2>/dev/null || echo "")
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LEROBOT_HOME="/data/beegfs/home/saifi/.cache/huggingface/lerobot"

# JAX/XLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

# W&B (train.py uses project="openpi" hardcoded)
WANDB_KEY_FILE="${HOME}/.config/wandb/api_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE}")
elif [[ -f "${HOME}/.netrc" ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=offline
fi

# ─── Pre-flight (system commands - before LD_LIBRARY_PATH) ───────────────────
cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Training π0-FAST LoRA (50 demos)"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  HF_TOKEN: ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "  W&B:      project=openpi (hardcoded in train.py)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

nvidia-smi
echo ""

# Clean any broken dataset cache
rm -rf /data/beegfs/home/saifi/.cache/huggingface/datasets/parquet/default-*/*.incomplete 2>/dev/null || true

# ─── Launch Training ─────────────────────────────────────────────────────────
# Do NOT export LD_LIBRARY_PATH — ptxas needs clean system env.
# Main process gets sysroot libs via --library-path.
# Use --num-workers=0 to avoid spawning child processes.

${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
    --library-path "${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib" \
    ${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --num-workers=0

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0-FAST Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
