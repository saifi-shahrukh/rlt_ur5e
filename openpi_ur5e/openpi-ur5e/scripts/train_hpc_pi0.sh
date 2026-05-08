#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# SLURM job: Fine-tune π0 LoRA on HPC cluster (V100 32GB)
# Hardware: 4x Tesla V100 SXM2 32GB | 187GB RAM | 56 Cores
# Container: build_env_310_final.sqsh (Python 3.10 + JAX + OpenPI)
# ═══════════════════════════════════════════════════════════════════

#SBATCH --job-name=pi0_peg_insertion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1x V100 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/pi0_peg_%j.out
#SBATCH --error=/data/beegfs/home/saifi/pi0_peg_%j.err

# Environment
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=offline

echo "═══════════════════════════════════════════════════════"
echo "  Job started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Config: pi0_ur5e_peg_insertion_lora (batch=16, V100 32GB)"
echo "═══════════════════════════════════════════════════════"

srun \
  --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
  --container-mount-home \
  --container-mounts=/data/beegfs/home/saifi:/data/beegfs/home/saifi \
  --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all \
  bash -c "cd /data/beegfs/home/saifi/openpi-ur5e && \
    uv run scripts/train.py pi0_ur5e_peg_insertion_lora \
      --exp-name=peg_insertion_hpc \
      --overwrite \
      --batch-size=16"

echo "Job finished: $(date)"
