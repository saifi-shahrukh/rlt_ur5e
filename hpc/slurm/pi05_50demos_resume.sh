#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: RESUME π0.5 LoRA (if needed - currently COMPLETE at step 4999)
# This is a safety script in case π0.5 needs re-running
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi05_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi05_resume_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi05_resume_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
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

# JAX/XLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
export JAX_TRACEBACK_FILTERING=off

# W&B — resume the existing run (wandb_id.txt has the run ID)
WANDB_KEY_FILE="${HOME}/.config/wandb/api_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE}")
elif [[ -f "${HOME}/.netrc" ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
fi
# If no API key available, fall back to offline
[[ -z "${WANDB_API_KEY:-}" ]] && export WANDB_MODE=offline

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${OPENPI}"

CHKPT_DIR="${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}"
echo "═══════════════════════════════════════════════════════════════"
echo "  RESUMING π0.5 LoRA"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"

echo "  Existing checkpoints:"
ls -d ${CHKPT_DIR}/[0-9]* 2>/dev/null | while read d; do echo "    ✓ step $(basename $d)"; done
echo ""
nvidia-smi
echo ""

# ─── Resume Training ─────────────────────────────────────────────────────────
# --resume flag tells OpenPI to load from latest checkpoint
${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --resume \
    --batch-size=4 \
    --grad-accumulation-steps=2 \
    --num-workers=4 \
    --num-train-steps=5000 \
    --save-interval=1000

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0.5 Training COMPLETE! $(date)"
echo "  Final checkpoint: ${CHKPT_DIR}/"
echo "═══════════════════════════════════════════════════════════════"
