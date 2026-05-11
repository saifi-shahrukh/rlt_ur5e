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
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi0_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_9demos"

# ─── Environment ──────────────────────────────────────────────────────────────
# PATH for system commands (nvidia-smi, hostname, etc.)
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
export CONDA_PREFIX="${VENV}"
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export WANDB_PROJECT="rlt-ur5e"
export WANDB_RUN_GROUP="hpc-pi0"
export WANDB_NAME="pi0_peg_9demos_$(date +%m%d_%H%M)"

# The Python runner: uses sysroot's ld-linux to load ALL libs through glibc 2.28
# This avoids any glibc version mixing between system loader and sysroot libs.
RUN_PYTHON="${SYSROOT}/lib64/ld-linux-x86-64.so.2 --library-path ${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib ${VENV}/bin/python3.11"

# ─── Run ─────────────────────────────────────────────────────────────────────
cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Training π0 LoRA"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Python:   $(${RUN_PYTHON} --version 2>&1)"
echo "  W&B:      project=${WANDB_PROJECT} | run=${WANDB_NAME}"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

nvidia-smi
echo ""

# Run training through sysroot loader (glibc 2.28 for everything)
${RUN_PYTHON} scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0 Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
