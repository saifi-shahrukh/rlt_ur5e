#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0.5 LoRA (9 demos peg insertion)
# Config: pi05_ur5e_peg_insertion_lora | batch_size=16 | 30k steps | ~3h
# π0.5 = improved architecture with cross-attention VLM ↔ action expert
# Norm stats: already in repo ✓
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi05_peg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi05_peg_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi05_peg_%j.err

set -e

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
CONFIG="pi05_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_9demos"

# ─── Environment ─────────────────────────────────────────────────────────────
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"

# ─── W&B Online Logging ──────────────────────────────────────────────────────
export WANDB_PROJECT="rlt-ur5e"
export WANDB_RUN_GROUP="hpc-pi05"
export WANDB_NAME="pi05_peg_9demos_$(date +%m%d_%H%M)"
# WANDB_API_KEY should be set via `wandb login` or exported in env

# ─── Run ─────────────────────────────────────────────────────────────────────
cd "${OPENPI}"
source .venv/bin/activate

echo "═══════════════════════════════════════════════════════════════"
echo "  Training π0.5 LoRA"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Python:   $(python --version)"
echo "  W&B:      project=${WANDB_PROJECT} | run=${WANDB_NAME}"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

nvidia-smi
echo ""

python scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0.5 Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
