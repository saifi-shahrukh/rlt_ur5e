#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: RESUME π0-FAST LoRA training
# Uses --resume flag to continue from latest checkpoint.
# W&B will resume the existing run (reads wandb_id.txt from checkpoint dir).
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi0fast_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi0fast_resume_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi0fast_resume_%j.err

set -euo pipefail

# ─── Config ───────────────────────────────���──────────────────────────────────
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

# JAX/XLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"
export JAX_TRACEBACK_FILTERING=off

# W&B — resume the existing run (reads wandb_id.txt)
WANDB_KEY_FILE="${HOME}/.config/wandb/api_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE}")
elif [[ -f "${HOME}/.netrc" ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
fi
[[ -z "${WANDB_API_KEY:-}" ]] && export WANDB_MODE=offline

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${OPENPI}"

CHKPT_DIR="${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}"
echo "═══════════════════════════════════════════════════════════════"
echo "  RESUMING π0-FAST LoRA"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"

echo "  Existing checkpoints:"
ls -d ${CHKPT_DIR}/[0-9]* 2>/dev/null | sort -V | tail -5 | while read d; do echo "    ✓ step $(basename $d)"; done
echo ""

if [[ ! -f "${CHKPT_DIR}/wandb_id.txt" ]]; then
    echo "  ⚠ wandb_id.txt not found - creating for offline mode"
    echo "offline_resume_$(date +%s)" > "${CHKPT_DIR}/wandb_id.txt"
fi

nvidia-smi
echo ""

# ─── Resume Training ─────────────────────────────────────────────────────────
${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --resume \
    --num-workers=8

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0-FAST Training Complete/Resumed! $(date)"
echo "  Checkpoint: ${CHKPT_DIR}/"
echo "═══════════════════════════════════════════════════════════════"
