#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0.5 LoRA (50 demos peg insertion)
# Config: pi05_ur5e_peg_insertion_lora | V100 32GB optimized
# Steps: 5000 (sufficient for LoRA on 50 demos)
# Note: batch=4 + grad_accum=2 to avoid OOM rematerialization
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi05_50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi05_50_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi05_50_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi05_ur5e_peg_insertion_lora"
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
# Lower mem fraction for pi0.5 — needs breathing room for rematerialization
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
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
echo "  Training π0.5 LoRA — 50 demos | V100 32GB"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Batch:    4 | Grad Accum: 2 (eff=8) | Workers: 4 | Steps: 5000"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
nvidia-smi
echo ""

# ─── Launch Training ─────────────────────────────────────────────────────────
# batch=4 + grad_accum=2: effective batch=8, avoids OOM rematerialization
# pi0.5 params=7.21 GiB + optimizer=5.22 GiB = 12.43 GiB fixed
# batch=4 activations ~9 GiB → total ~21 GiB (safe in 32 GiB)
# batch=8 activations ~19 GiB → total ~31 GiB (triggers rematerialization!)
${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --batch-size=4 \
    --grad-accumulation-steps=2 \
    --num-workers=4 \
    --num-train-steps=5000 \
    --save-interval=1000

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0.5 Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
