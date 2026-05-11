#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0 LoRA (9 demos peg insertion)
# Config: pi0_ur5e_peg_insertion_lora | batch_size=4 | 30k steps
# Norm stats: pre-computed in repo ✓
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi0_peg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi0_peg_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi0_peg_%j.err

set -e

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
CONFIG="pi0_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_9demos"

# ─── Environment (uses sysroot glibc 2.28) ───────────────────────────────────
source "${VENV}/activate_hpc.sh"

# ─── W&B Online Logging ──────────────────────────────────────────────────────
export WANDB_RUN_GROUP="hpc-pi0"
export WANDB_NAME="pi0_peg_9demos_$(date +%m%d_%H%M)"
# API key set via ~/.bashrc or uncomment below:
# export WANDB_API_KEY="your-key-here"

# ─── Run ─────────────────────────────────────────────────────────────────────
cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Training π0 LoRA"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Python:   $(python --version)"
echo "  glibc:    $(python -c \"import ctypes; libc=ctypes.CDLL('libc.so.6'); f=libc.gnu_get_libc_version; f.restype=ctypes.c_char_p; print(f().decode())\" 2>/dev/null || echo 'unknown')"
echo "  W&B:      project=${WANDB_PROJECT} | run=${WANDB_NAME}"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

nvidia-smi
echo ""

run_python scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0 Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
