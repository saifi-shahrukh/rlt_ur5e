#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM: Compute Normalization Stats for π0.5
# (π0 stats already exist in repo, only π0.5 needs computing)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=norm_stats
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/beegfs/home/saifi/logs/norm_stats_%j.out
#SBATCH --error=/data/beegfs/home/saifi/logs/norm_stats_%j.err

set -e

OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
cd "${OPENPI}"
source .venv/bin/activate

export HF_HOME="/data/beegfs/home/saifi/.cache/huggingface"

echo "═══════════════════════════════════════════════════════════════"
echo "  Computing Normalization Stats"
echo "  Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Time: $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# π0 stats already exist in repo, but compute anyway to be safe
echo "[1/2] pi0_ur5e_peg_insertion_lora..."
python scripts/compute_norm_stats.py --config-name=pi0_ur5e_peg_insertion_lora
echo "  ✓ Done"
echo ""

# π0.5 stats DO NOT EXIST — must compute
echo "[2/2] pi05_ur5e_peg_insertion_lora..."
python scripts/compute_norm_stats.py --config-name=pi05_ur5e_peg_insertion_lora
echo "  ✓ Done"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ All norm stats computed! $(date)"
echo "═══════════════════════════════════════════════════════════════"
