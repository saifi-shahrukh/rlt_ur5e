# HPC Cluster Training Guide

> Fine-tune π0 and π0.5 models on V100 32GB GPUs (exceeds local 16GB capacity).

## Cluster Info

| Field | Value |
|-------|-------|
| Head Node | `hpc-headnode.iis.fhg.de` |
| SSH | `ssh -x saifi@hpc-headnode.iis.fhg.de` |
| GPUs | 4× Tesla V100 SXM2 32GB |
| Storage | `/data/beegfs/home/saifi/` |
| Scheduler | SLURM |
| Container | enroot (squashfs) |

---

## Initial Setup (One-Time)

### 1. Clone RLT repo on HPC

```bash
ssh -x saifi@hpc-headnode.iis.fhg.de
cd /data/beegfs/home/saifi/
git clone https://github.com/saifi-shahrukh/rlt_ur5e.git
cd rlt_ur5e
```

### 2. Transfer Dataset

```bash
# From LOCAL machine:
rsync -avz --progress \
    ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual/ \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual/
```

### 3. Create Dataset Symlink on HPC

```bash
mkdir -p ~/.cache/huggingface/lerobot/saifi/
ln -sf /data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual
```

### 4. Build Container (or use existing)

If container already exists:
```bash
ls /data/beegfs/home/saifi/build_env_310_final.sqsh
```

If building fresh:
```bash
# Import base image
enroot import docker://python:3.11-slim-bullseye
enroot create --name openpi_env python+3.11-slim-bullseye.sqsh

# Install inside container
enroot start --root --rw openpi_env bash
apt update && apt install -y build-essential git curl ffmpeg libsm6 libxext6 pkg-config linux-libc-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e
uv pip install --system -e .
exit

# Export
enroot export -f -o /data/beegfs/home/saifi/build_env_310_final.sqsh openpi_env
```

### 5. Create venv alternative (if no enroot)

```bash
cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
# Or with uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## Training Commands

### Compute Norm Stats (required before first training)

```bash
srun --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
     --container-mount-home \
     --container-mounts=/data/beegfs/home/saifi:/data/beegfs/home/saifi \
     --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all \
     -p gpu --gres=gpu:1 --mem=32G --cpus-per-task=8 \
     bash -c "cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e && \
       .venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_ur5e_peg_insertion_lora && \
       .venv/bin/python scripts/compute_norm_stats.py --config-name=pi05_ur5e_peg_insertion_lora"
```

### Train π0 LoRA (9 demos)

```bash
sbatch scripts/train_hpc_pi0.sh
```

**Script: `scripts/train_hpc_pi0.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=pi0_peg_9demos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/beegfs/home/saifi/pi0_peg_%j.out
#SBATCH --error=/data/beegfs/home/saifi/pi0_peg_%j.err

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=offline

srun \
  --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
  --container-mount-home \
  --container-mounts=/data/beegfs/home/saifi:/data/beegfs/home/saifi \
  --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all \
  bash -c "cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e && \
    .venv/bin/python scripts/train.py pi0_ur5e_peg_insertion_lora \
      --exp-name=peg_insertion_9demos \
      --overwrite"
```

### Train π0.5 LoRA (9 demos)

```bash
sbatch scripts/train_hpc_pi05.sh
```

**Script: `scripts/train_hpc_pi05.sh`**
```bash
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
```

### Monitor Jobs

```bash
squeue -u saifi                    # Check job status
tail -f /data/beegfs/home/saifi/pi0_peg_*.out  # Check output
scancel <JOB_ID>                   # Cancel job
```

### Download Checkpoints to Local

```bash
# From LOCAL machine:
rsync -avz --progress \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/ \
    ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/
```

---

## Expected Training Times

| Model | GPU | Steps | ETA |
|-------|-----|-------|-----|
| π0-FAST LoRA | Local RTX 5070 Ti | 30k | ~4h |
| π0 LoRA | V100 32GB | 30k | ~2h |
| π0.5 LoRA | V100 32GB | 30k | ~3h |
