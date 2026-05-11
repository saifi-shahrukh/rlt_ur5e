#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Train π0-FAST LoRA (9 demos peg insertion)
# Config: pi0_fast_ur5e_peg_insertion_lora | batch_size=16 | 30k steps
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=pi0f_peg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/pi0fast_peg_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/pi0fast_peg_%j.err

set -e

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
CONFIG="pi0_fast_ur5e_peg_insertion_lora"
EXP_NAME="peg_insertion_9demos"

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
# W&B: read API key from netrc (multi-line format) or fallback to env
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
echo "  Training π0-FAST LoRA"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  HF_TOKEN: ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "  Offline:  HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "  W&B:      project=openpi (hardcoded in train.py)"
echo "  Start:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

nvidia-smi
echo ""

# Clean any broken dataset cache from previous failed runs
rm -rf /data/beegfs/home/saifi/.cache/huggingface/datasets/parquet/default-*/*.incomplete 2>/dev/null || true

# Patchelf ptxas/nvlink to use sysroot ld-linux (same approach as Python).
# When LD_LIBRARY_PATH has sysroot libs, system binaries crash because
# system ld-linux 2.17 can't load sysroot libc 2.28. Patchelf makes them
# use sysroot ld-linux 2.28 directly, so they work with any LD_LIBRARY_PATH.
PATCHELF="${VENV}/bin/patchelf"
if [[ -f "${VENV}/bin/ptxas.real" ]]; then
    # Only patchelf once (check if interpreter is already sysroot)
    CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${VENV}/bin/ptxas.real" 2>/dev/null || echo "")
    if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
        ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" \
                   --set-rpath "${SYSROOT}/lib64:${SYSROOT}/usr/lib64" \
                   "${VENV}/bin/ptxas.real"
    fi
    rm -f "${VENV}/bin/ptxas"
    ln -sf ptxas.real "${VENV}/bin/ptxas"
    echo "ptxas: $(${VENV}/bin/ptxas --version 2>&1 | grep release || echo 'FAILED')"
fi
if [[ -f "${VENV}/bin/nvlink.real" ]]; then
    CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${VENV}/bin/nvlink.real" 2>/dev/null || echo "")
    if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
        ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" \
                   --set-rpath "${SYSROOT}/lib64:${SYSROOT}/usr/lib64" \
                   "${VENV}/bin/nvlink.real"
    fi
    rm -f "${VENV}/bin/nvlink"
    ln -sf nvlink.real "${VENV}/bin/nvlink"
    echo "nvlink: $(${VENV}/bin/nvlink --version 2>&1 | grep release || echo 'FAILED')"
fi
# Also patchelf pip-installed ptxas/nvlink (XLA checks these paths too)
PIP_CUDA="${VENV}/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin"
if [[ -f "${PIP_CUDA}/ptxas" ]] && [[ ! -L "${PIP_CUDA}/ptxas" ]]; then
    CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${PIP_CUDA}/ptxas" 2>/dev/null || echo "")
    if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
        ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" \
                   --set-rpath "${SYSROOT}/lib64:${SYSROOT}/usr/lib64" \
                   "${PIP_CUDA}/ptxas"
    fi
fi
if [[ -f "${PIP_CUDA}/nvlink" ]] && [[ ! -L "${PIP_CUDA}/nvlink" ]]; then
    CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${PIP_CUDA}/nvlink" 2>/dev/null || echo "")
    if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
        ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" \
                   --set-rpath "${SYSROOT}/lib64:${SYSROOT}/usr/lib64" \
                   "${PIP_CUDA}/nvlink"
    fi
fi

# ─── Launch Training ─────────────────────────────────────────────────────────
# Set LD_LIBRARY_PATH NOW (after all system commands are done).
# This is inherited by multiprocessing worker subprocesses so they
# also load sysroot glibc 2.28 libs (fixes librt.so.1 __clock_nanosleep error).
export LD_LIBRARY_PATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib:${LD_LIBRARY_PATH:-}"

# Launch through sysroot's ld-linux (main process uses --library-path,
# spawned workers inherit LD_LIBRARY_PATH — both get glibc 2.28)
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
