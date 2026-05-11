#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0-FAST LoRA (50 demos peg insertion)
# Config: pi0_fast_ur5e_peg_insertion_lora | batch_size=1 | 30k steps
# Dataset: saifi/ur5e-peg-insertion-50demos-v2
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

# Don't use set -e: patchelf failures shouldn't kill the job
# set -e

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi0_fast_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_50demos_v2"

# ─── Environment (system commands work fine here - no LD_LIBRARY_PATH yet) ───
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
export CONDA_PREFIX="${VENV}"

# HuggingFace: token for gated models + offline mode (dataset is local)
export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export HF_TOKEN=$(cat /data/beegfs/home/saifi/.cache/huggingface/token 2>/dev/null || echo "")
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LEROBOT_HOME="/data/beegfs/home/saifi/.cache/huggingface/lerobot"

# JAX/XLA
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

# W&B (train.py uses project="openpi" hardcoded)
WANDB_KEY_FILE="${HOME}/.config/wandb/api_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY=$(cat "${WANDB_KEY_FILE}")
elif [[ -f "${HOME}/.netrc" ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=offline
fi

# ─── Pre-flight (system commands - before LD_LIBRARY_PATH) ───────────────────
cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Training π0-FAST LoRA (50 demos)"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  HF_TOKEN: ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "  W&B:      project=openpi (hardcoded in train.py)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

nvidia-smi
echo ""

# Clean any broken dataset cache
rm -rf /data/beegfs/home/saifi/.cache/huggingface/datasets/parquet/default-*/*.incomplete 2>/dev/null || true

# ─── Fix ptxas/nvlink (patchelf to use sysroot ld-linux) ─────────────────────
# ptxas/nvlink from cuda-nvcc conda package need glibc 2.28.
# When LD_LIBRARY_PATH has sysroot, system-linked binaries crash.
# Patchelf makes them use sysroot's ld-linux directly.
PATCHELF="${VENV}/bin/patchelf"

fix_binary() {
    local BINARY="$1"
    if [[ ! -f "${BINARY}" ]] || [[ -L "${BINARY}" ]]; then
        return 0
    fi
    local CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${BINARY}" 2>/dev/null || echo "")
    if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
        echo "  Patching: ${BINARY}"
        ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" \
                   --set-rpath "${SYSROOT}/lib64:${SYSROOT}/usr/lib64" \
                   "${BINARY}" 2>/dev/null || echo "  WARNING: patchelf failed for ${BINARY}"
    fi
}

# Patch conda-installed binaries
if [[ -f "${VENV}/bin/ptxas" ]]; then
    fix_binary "${VENV}/bin/ptxas"
    echo "  ptxas: $(${VENV}/bin/ptxas --version 2>&1 | grep -i release || echo 'version check failed')"
fi
if [[ -f "${VENV}/bin/nvlink" ]]; then
    fix_binary "${VENV}/bin/nvlink"
fi

# Also patch pip-installed binaries (XLA checks these paths)
PIP_CUDA="${VENV}/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin"
if [[ -d "${PIP_CUDA}" ]]; then
    for bin in ptxas nvlink; do
        [[ -f "${PIP_CUDA}/${bin}" ]] && fix_binary "${PIP_CUDA}/${bin}"
    done
fi

# ─── Launch Training ─────────────────────────────────────────────────────────
# Set LD_LIBRARY_PATH NOW (after all system commands and patchelf are done).
# Workers inherit this so they also get glibc 2.28 libs.
export LD_LIBRARY_PATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib:${LD_LIBRARY_PATH:-}"

# Launch through sysroot's ld-linux
${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
    --library-path "${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib" \
    ${VENV}/bin/python3.11 scripts/train.py ${CONFIG} \
    --exp-name=${EXP_NAME} \
    --overwrite

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ π0-FAST Training Complete! $(date)"
echo "  Checkpoint: ${OPENPI}/checkpoints/${CONFIG}/${EXP_NAME}/"
echo "═══════════════════════════════════════════════════════════════"
