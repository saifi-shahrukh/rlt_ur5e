#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Compute norm stats for v2 configs (2-camera)
# Must be run BEFORE v2 training
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=norm_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/norm_stats_v2_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/norm_stats_v2_%j.err

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${VENV}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"
export CONDA_PREFIX="${VENV}"

export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"
export HF_TOKEN=$(cat /data/beegfs/home/saifi/.cache/huggingface/token 2>/dev/null || echo "")
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LEROBOT_HOME="/data/beegfs/home/saifi/.cache/huggingface/lerobot"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

# ─── Pre-flight ──────────────────────────────────────────────────────────────
cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Computing Norm Stats for v2 Configs (2-camera)"
echo "  Node: $(hostname)"
echo "  Start: $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Compute for all 3 v2 configs
for CONFIG in pi0_ur5e_peg_insertion_v2_lora pi0_fast_ur5e_peg_insertion_v2_lora pi05_ur5e_peg_insertion_v2_lora; do
    echo "  → Computing norm stats for: ${CONFIG}"
    PYTHONUNBUFFERED=1 ${VENV}/bin/python3.11 scripts/compute_norm_stats.py --config-name=${CONFIG}
    echo "  ✓ Done: ${CONFIG}"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ All norm stats computed! $(date)"
echo "  Now run: bash hpc/05_train_v2.sh all"
echo "═══════════════════════════════════════════════════════════════"
