#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0 LoRA (50 demos peg insertion)
# Config: pi0_ur5e_peg_insertion_lora | V100 32GB optimized
# Steps: 5000 (sufficient for LoRA on 50 demos)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi0_50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi0_50_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi0_50_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi0_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_50demos"

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV}"

# HuggingFace
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export HF_TOKEN=$(cat /data/beegfs/home/saifi/.cache/huggingface/token 2>/dev/null || echo "")
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LEROBOT_HOME="/data/beegfs/home/saifi/.cache/huggingface/lerobot"

# JAX/XLA — optimized for V100 32GB
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
export JAX_TRACEBACK_FILTERING=off

# W&B
WANDB_KEY_FILE="${HOME}/.config/wandb/api_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE}")
elif [[ -f "${HOME}/.netrc" ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
fi
[[ -z "${WANDB_API_KEY:-}" ]] && export WANDB_MODE=offline

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Training π0 LoRA — 50 demos | V100 32GB"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Batch:    8 | Workers: 4 | Steps: 5000"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
nvidia-smi
echo ""

# ─── Launch Training ─────────────────────────────────────────────────────────
# batch=8: safe memory (~8 GiB activations), no rematerialization
# workers=4: parallel data loading (DT_RPATH provides sysroot libs)
# 5000 steps: sufficient for LoRA convergence on 50 demos
# save_interval=1000: checkpoints at 1k, 2k, 3k, 4k, 5k
${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --batch-size=8 \
    --num-workers=4 \
    --num-train-steps=5000 \
    --save-interval=1000

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0 Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
