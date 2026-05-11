#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0.5 LoRA (50 demos peg insertion)
# Config: pi05_ur5e_peg_insertion_lora | batch_size=16 | 30k steps
# Dataset: saifi/ur5e-peg-insertion-50demos-v2
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

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi05_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_50demos_v2"

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
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
echo "  Training π0.5 LoRA (50 demos)"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
nvidia-smi
echo ""

rm -rf /data/beegfs/home/saifi/.cache/huggingface/datasets/parquet/default-*/*.incomplete 2>/dev/null || true

# ─── Fix ptxas/nvlink ────────────────────────────────────────────────────────
PATCHELF="${VENV}/bin/patchelf"
fix_binary() {
    local BINARY="$1"
    [[ ! -f "${BINARY}" ]] || [[ -L "${BINARY}" ]] && return 0
    local INTERP=$(${PATCHELF} --print-interpreter "${BINARY}" 2>/dev/null || echo "")
    if [[ "${INTERP}" != *"sysroot"* ]]; then
        ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" \
                   --set-rpath "${SYSROOT}/lib64:${SYSROOT}/usr/lib64" \
                   "${BINARY}" 2>/dev/null || true
    fi
}
[[ -f "${VENV}/bin/ptxas" ]] && fix_binary "${VENV}/bin/ptxas"
[[ -f "${VENV}/bin/nvlink" ]] && fix_binary "${VENV}/bin/nvlink"
PIP_CUDA="${VENV}/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin"
[[ -f "${PIP_CUDA}/ptxas" ]] && fix_binary "${PIP_CUDA}/ptxas"
[[ -f "${PIP_CUDA}/nvlink" ]] && fix_binary "${PIP_CUDA}/nvlink"

# ─── Launch Training ─────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib:${LD_LIBRARY_PATH:-}"

${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
    --library-path "${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib" \
    ${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite

echo "✓ π0.5 Training Complete! $(date)"
echo "Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
