#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: RESUME π0 LoRA from step 4000 → 5000
# Only ~360 steps remaining (~56 minutes)
# KEY: No --overwrite flag! This resumes from latest checkpoint.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi0_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi0_resume_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi0_resume_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
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

# JAX/XLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
export JAX_TRACEBACK_FILTERING=off

# W&B — for resume, we start a NEW run (old one is marked finished)
# Using offline mode avoids "resume=must" failures
export WANDB_MODE=offline
# Still set key in case we switch to online later
WANDB_KEY_FILE="${HOME}/.config/wandb/api_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE}")
elif [[ -f "${HOME}/.netrc" ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
fi

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${OPENPI}"

CHKPT_DIR="${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}"
echo "═══════════════════════════════════════════════════════════════"
echo "  RESUMING π0 LoRA — step 4000 → 5000"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Show existing checkpoints
echo "  Existing checkpoints:"
ls -d ${CHKPT_DIR}/[0-9]* 2>/dev/null | while read d; do echo "    ✓ step $(basename $d)"; done
echo ""

# Verify wandb_id.txt exists (required for --resume with W&B)
if [[ ! -f "${CHKPT_DIR}/wandb_id.txt" ]]; then
    echo "  ⚠ wandb_id.txt not found - creating dummy for offline mode"
    echo "offline_resume_$(date +%s)" > "${CHKPT_DIR}/wandb_id.txt"
fi
echo "  W&B run ID: $(cat ${CHKPT_DIR}/wandb_id.txt)"
echo ""

nvidia-smi
echo ""

# ─── Resume Training ─────────────────────────────────────────────────────────
# --resume flag tells OpenPI to load from latest checkpoint (step 4000)
# Cannot use --overwrite with --resume (they conflict)
${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --resume \
    --batch-size=8 \
    --num-workers=4 \
    --num-train-steps=5000 \
    --save-interval=1000

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0 Training COMPLETE! $(date)"
echo "  Final checkpoint: ${CHKPT_DIR}/5000/"
echo "═══════════════════════════════════════════════════════════════"
