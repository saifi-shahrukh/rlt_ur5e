# 🖥️ HPC Cluster Training — π0 and π0.5 Fine-Tuning

> For models that exceed local GPU capacity (16GB), use the HPC cluster with V100 32GB GPUs.

---

## Cluster Specifications

| Component | Details |
|-----------|--------|
| Head Node | `hpc-headnode.iis.fhg.de` |
| SSH Access | `ssh -x saifi@hpc-headnode.iis.fhg.de` |
| GPUs | 4× Tesla V100 SXM2 **32GB** each |
| RAM | 187 GB |
| CPUs | 56 Cores |
| Storage | BeeGFS: `/data/beegfs/home/saifi/` |
| Scheduler | SLURM |
| Container | enroot (squashfs images) |
| WandB | https://wandb.ai/saifi/openpi |

---

## Training Strategy Overview

| Model | Where | VRAM | Batch | Rank | ETA |
|-------|-------|------|-------|------|-----|
| **π0-FAST LoRA** | Local (RTX 5070 Ti 16GB) | 15.7 GB | 1 | 4 | ~4h |
| **π0 LoRA** | **HPC** (V100 32GB) | ~24 GB | 16 | 16 | ~2h |
| **π0.5 LoRA** | **HPC** (V100 32GB) | ~28 GB | 16 | 16 | ~3h |

---

## Complete Setup Guide (From Scratch)

### Step 1: SSH into HPC

```bash
ssh -x saifi@hpc-headnode.iis.fhg.de
```

### Step 2: Transfer Dataset from Local Machine to HPC

```bash
# Run from your LOCAL machine
rsync -avz --progress \
    ~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual/ \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual/
```

### Step 3: Transfer OpenPI Code to HPC

```bash
# Run from your LOCAL machine
rsync -avz --progress \
    --exclude='.venv' \
    --exclude='checkpoints' \
    --exclude='wandb' \
    --exclude='__pycache__' \
    --exclude='.git' \
    ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e/ \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/openpi-ur5e/
```

### Step 4: Build Container on HPC (One-Time Setup)

```bash
ssh -x saifi@hpc-headnode.iis.fhg.de

# Import base Python image
enroot import docker://python:3.11-slim-bullseye
enroot create --name openpi_env python+3.11-slim-bullseye.sqsh

# Start container with root access for installs
enroot start --root --rw openpi_env bash
```

**Inside the container:**
```bash
# Install system dependencies
apt update && apt install -y \
    build-essential git curl ffmpeg libsm6 libxext6 pkg-config \
    linux-libc-dev

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Install OpenPI
cd /data/beegfs/home/saifi/openpi-ur5e
uv pip install --system -e .

# Verify
python -c "import jax; print('JAX:', jax.__version__); print('GPU:', jax.devices())"

# Exit container
exit
```

**Export container as squashfs image:**
```bash
enroot export -f -o /data/beegfs/home/saifi/build_env_310_final.sqsh openpi_env
```

### Step 5: Test Container Works

```bash
srun --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
     --container-mount-home \
     --container-mounts=/data/beegfs/home/saifi:/data/beegfs/home/saifi \
     --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all \
     -p gpu --gres=gpu:1 \
     python3 -c "import jax; print('JAX:', jax.__version__); \
                 print('Devices:', jax.devices()); \
                 import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 6: Create Dataset Symlink on HPC

```bash
ssh -x saifi@hpc-headnode.iis.fhg.de

# OpenPI expects datasets in HuggingFace cache location
mkdir -p ~/.cache/huggingface/lerobot/saifi/
ln -sf /data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual

# Verify
ls -la ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual/
```

### Step 7: Compute Normalization Stats on HPC

```bash
srun --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
     --container-mount-home \
     --container-mounts=/data/beegfs/home/saifi:/data/beegfs/home/saifi \
     --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all \
     -p gpu --gres=gpu:1 --mem=32G --cpus-per-task=8 \
     bash -c "cd /data/beegfs/home/saifi/openpi-ur5e && \
       uv run scripts/compute_norm_stats.py --config-name=pi0_ur5e_peg_insertion_lora && \
       uv run scripts/compute_norm_stats.py --config-name=pi05_ur5e_peg_insertion_lora"
```

### Step 8: Configure WandB

```bash
# Option A: Online mode (if HPC has internet)
srun --container-image=/data/beegfs/home/saifi/build_env_310_final.sqsh \
     --container-mount-home \
     -p gpu --gres=gpu:1 \
     bash -c "wandb login"

# Option B: Offline mode (sync later)
# Set WANDB_MODE=offline in SLURM scripts (already configured)
# After training, sync from local machine:
# wandb sync /data/beegfs/home/saifi/openpi-ur5e/wandb/run-*/
```

**WandB Project:** https://wandb.ai/saifi/openpi

---

## Submit Training Jobs

### Train π0 LoRA (V100 32GB)

```bash
cd /data/beegfs/home/saifi/openpi-ur5e
sbatch scripts/train_hpc_pi0.sh
```

<details>
<summary>📄 View scripts/train_hpc_pi0.sh</summary>

```bash
#!/bin/bash
#SBATCH --job-name=pi0_peg_insertion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1x V100 32GB
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
  bash -c "cd /data/beegfs/home/saifi/openpi-ur5e && \
    uv run scripts/train.py pi0_ur5e_peg_insertion_lora \
      --exp-name=peg_insertion_hpc \
      --overwrite"
```
</details>

### Train π0.5 LoRA (V100 32GB)

```bash
cd /data/beegfs/home/saifi/openpi-ur5e
sbatch scripts/train_hpc_pi05.sh
```

<details>
<summary>📄 View scripts/train_hpc_pi05.sh</summary>

```bash
#!/bin/bash
#SBATCH --job-name=pi05_peg_insertion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1x V100 32GB
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
  bash -c "cd /data/beegfs/home/saifi/openpi-ur5e && \
    uv run scripts/train.py pi05_ur5e_peg_insertion_lora \
      --exp-name=peg_insertion_hpc \
      --overwrite"
```
</details>

---

## Monitor Training on HPC

### Check Job Status
```bash
# See all your jobs
squeue -u ort

# Detailed job info
scontrol show job <JOBID>

# Cancel a job
scancel <JOBID>
```

### Watch Logs in Real-Time
```bash
# π0 training output
tail -f /data/beegfs/home/saifi/pi0_peg_*.out

# π0.5 training output
tail -f /data/beegfs/home/saifi/pi05_peg_*.out

# Check for errors
tail -f /data/beegfs/home/saifi/pi0_peg_*.err
```

### Check GPU Usage (interactive)
```bash
srun -p gpu --gres=gpu:1 --pty nvidia-smi
```

### WandB Monitoring

**If online mode:**
- Dashboard: https://wandb.ai/saifi/openpi
- Real-time loss curves, gradient norms, learning rate

**If offline mode (sync after training):**
```bash
# On HPC:
cd /data/beegfs/home/saifi/openpi-ur5e
wandb sync wandb/run-*/

# Or from local machine:
rsync -avz saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/openpi-ur5e/wandb/ ./wandb_hpc/
wandb sync wandb_hpc/run-*/
```

---

## Copy Checkpoints Back to Local Machine

After training completes, copy checkpoints for inference on the robot:

```bash
# From your LOCAL machine:

# π0 checkpoint:
rsync -avz --progress \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/openpi-ur5e/checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_hpc/30000/ \
    ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e/checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_hpc/30000/

# π0.5 checkpoint:
rsync -avz --progress \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/openpi-ur5e/checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_hpc/30000/ \
    ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e/checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_hpc/30000/
```

---

## Serve HPC-Trained Models Locally (for Robot Inference)

### Serve π0 (trained on HPC)
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_hpc/30000
```

### Serve π0.5 (trained on HPC)
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_hpc/30000
```

### Run on Robot (same for all models)
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --fps=30
```

---

## Model Comparison

| Aspect | π0-FAST (Local) | π0 (HPC) | π0.5 (HPC) |
|--------|----------------|----------|------------|
| Architecture | LoRA on Gemma 2B + FAST tokenizer | LoRA on Gemma 2B + Gemma 300M expert | LoRA + cross-attention VLM-expert |
| Action decoding | Discrete tokens (fast) | Continuous diffusion (slow) | Continuous diffusion (slow) |
| VRAM needed | 15.7 GB | ~24 GB | ~28 GB |
| Training time | ~4h (batch=1) | ~2h (batch=16) | ~3h (batch=16) |
| Inference speed | Fast (single forward pass) | Slow (iterative denoising) | Slow (iterative denoising) |
| Quality | Good | Better | Best |
| Best for | Quick iteration, testing | Final deployment | Maximum precision |

---

## Directory Structure on HPC

```
/data/beegfs/home/saifi/
├── openpi-ur5e/                    # Code (synced from local)
│   ├── src/openpi/training/config.py
│   ├── scripts/
│   │   ├── train.py
│   │   ├── train_hpc_pi0.sh        # SLURM job script
│   │   └── train_hpc_pi05.sh       # SLURM job script
│   ├── checkpoints/                # Training output
│   │   ├── pi0_ur5e_peg_insertion_lora/peg_insertion_hpc/
│   │   └── pi05_ur5e_peg_insertion_lora/peg_insertion_hpc/
│   ├── assets/                     # Norm stats
│   └── wandb/                      # WandB logs
├── datasets/
│   └── saifi/ur5e-peg-insertion-dual/  # Dataset (synced from local)
├── build_env_310_final.sqsh                # Container image
├── pi0_peg_*.out                   # SLURM stdout
└── pi0_peg_*.err                   # SLURM stderr
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `enroot: command not found` | `module load enroot` or check cluster docs |
| Container build fails | Check disk space: `df -h /data/local/enroot/` |
| Job stuck in PENDING | Check queue: `squeue -p gpu`, reduce `--time` |
| OOM on V100 | Reduce batch: edit SLURM script `--batch-size=8` |
| Dataset not found | Check symlink: `ls -la ~/.cache/huggingface/lerobot/saifi/` |
| Checkpoint incomplete | Check error log: `cat /data/beegfs/home/saifi/pi0_peg_*.err` |
| WandB not syncing | Use offline mode + sync after: `wandb sync wandb/run-*/` |
| `gs://` download fails | HPC may not have internet; pre-download checkpoint locally and rsync |
| Permission denied on container | Use `--container-mount-home` flag |
| SLURM time limit | Default is 12h; increase with `--time=24:00:00` if needed |

### Pre-Download Checkpoints (if HPC has no internet)

```bash
# On LOCAL machine (has internet), download base checkpoints:
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate
python -c "from openpi.training import download; download.maybe_download('gs://openpi-assets/checkpoints/pi0_base/params')"
python -c "from openpi.training import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params')"

# Transfer to HPC:
rsync -avz --progress \
    ~/.cache/openpi/openpi-assets/checkpoints/ \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/.cache/openpi/openpi-assets/checkpoints/
```

---

## Full Workflow (End-to-End)

```bash
# ═══════════════════════════════════════════════════════════
# 1. Record demos LOCALLY (on robot workstation)
# ═══════════════════════════════════════════════════════════
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate
./scripts/record_wrapper.sh \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --robot.freedrive=False \
    --teleop.type=keyboard_ur5e --teleop.robot_ip=172.22.1.139 --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.num_episodes=20 --dataset.fps=30 --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False

# ═══════════════════════════════════════════════════════════
# 2. Transfer data to HPC
# ═══════════════════════════════════════════════════════════
rsync -avz ~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual/ \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual/

rsync -avz --exclude='.venv' --exclude='checkpoints' --exclude='wandb' \
    ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e/ \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/openpi-ur5e/

# ═══════════════════════════════════════════════════════════
# 3. Submit training job on HPC
# ═══════════════════════════════════════════════════════════
ssh -x saifi@hpc-headnode.iis.fhg.de
cd /data/beegfs/home/saifi/openpi-ur5e
sbatch scripts/train_hpc_pi0.sh     # π0
sbatch scripts/train_hpc_pi05.sh    # π0.5
squeue -u ort                       # check status
tail -f /data/beegfs/home/saifi/pi0_peg_*.out  # watch progress
exit

# ═══════════════════════════════════════════════════════════
# 4. Copy checkpoint back LOCAL after training completes
# ═══════════════════════════════════════════════════════════
rsync -avz saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/openpi-ur5e/checkpoints/ \
    ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e/checkpoints/

# ═══════════════════════════════════════════════════════════
# 5. Serve & deploy on robot
# ═══════════════════════════════════════════════════════════
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_hpc/30000
```
