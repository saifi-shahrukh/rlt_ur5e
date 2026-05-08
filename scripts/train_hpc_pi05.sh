#!/bin/bash
#SBATCH --job-name=pi05_peg_9demos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/pi05_peg_%j.out
#SBATCH --error=/data/beegfs/home/saifi/pi05_peg_%j.err

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=offline

srun \
  --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
  --container-mount-home \
  --container-mounts=/data/beegfs/home/saifi:/data/beegfs/home/saifi \
  --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all \
  bash -c "cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e && \
    .venv/bin/python scripts/train.py pi05_ur5e_peg_insertion_lora \
      --exp-name=peg_insertion_9demos \
      --overwrite"
