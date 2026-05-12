#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0-FAST LoRA (50 demos peg insertion)
# Config: pi0_fast_ur5e_peg_insertion_lora | V100 32GB
# Steps: 30,000 (official OpenPI standard for LoRA fine-tuning)
# LoRA: rank=16 (same as pi0 — standard for HPC training)
# NOTE: 12hr SLURM limit may not be enough. Use --resume to continue.
# REQUIRES: Run hpc/cache_fast_tokenizer.sh on headnode first!
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

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
CONFIG="pi0_fast_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_50demos"

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV}"

# HuggingFace — FAST tokenizer must be pre-cached!
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export HF_TOKEN=$(cat /data/beegfs/home/saifi/.cache/huggingface/token 2>/dev/null || echo "")
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LEROBOT_HOME="/data/beegfs/home/saifi/.cache/huggingface/lerobot"
export TRANSFORMERS_OFFLINE=1

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
echo "  Training π0-FAST LoRA — 50 demos | V100 32GB"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Steps:    30,000 | Batch: 8 | LoRA rank: 16 | Save: every 500"
echo "  WARNING:  12hr limit. Will need --resume if killed."
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
nvidia-smi
echo ""

# ─── Launch Training ─────────────────────────────────────────────────────────
# Config defaults: batch=8, save_interval=500, num_train_steps=30000, rank=16
# num_workers=8: max parallel data loading (matches cpus-per-task=8)
${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --num-workers=8

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0-FAST Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
