# RLT UR5e — Robot Learning & Training for UR5e Manipulation

Fine-tune Physical Intelligence's **OpenPI** foundation models (π0, π0.5, π0-FAST)
for UR5e robot manipulation tasks using LoRA on the Fraunhofer IIS HPC cluster.

## Project Structure

    rlt_ur5e/
    ├── README.md                           # This file
    ├── hpc/                                # HPC training scripts & docs
    │   ├── README.md                       # Quick reference
    │   ├── HPC_SETUP_README.md             # Detailed setup & troubleshooting
    │   ├── ADDING_NEW_DATASET.md           # Guide for new datasets
    │   ├── setup_hpc_env.sh                # One-time environment setup
    │   ├── 03_train.sh                     # Submit training jobs
    │   ├── 04_status.sh                    # Monitor jobs
    │   ├── 05_download_checkpoints.sh      # Download trained models
    │   └── slurm/                          # SLURM job scripts
    │       ├── pi0.sh                      # π0 training
    │       ├── pi05.sh                     # π0.5 training
    │       └── pi0_fast.sh                 # π0-FAST training
    ├── openpi_ur5e/                        # OpenPI framework (submodule)
    │   └── openpi-ur5e/
    │       ├── src/openpi/                 # Model & training code
    │       ├── scripts/train.py            # Training entry point
    │       ├── assets/                     # Pre-computed norm stats
    │       ├── pyproject.toml              # Package definition
    │       └── uv.lock                     # Pinned dependencies
    └── update_on_readme.md                 # Development notes

## Models

| Model | Architecture | Params (LoRA) | VRAM | Training Time |
|-------|-------------|---------------|------|---------------|
| **π0** | Flow matching + VLM | ~50M trainable | ~16 GB | ~8-10 hrs |
| **π0.5** | + Cross-attention | ~50M trainable | ~28 GB | ~10-12 hrs |
| **π0-FAST** | Continuous action tokens | ~50M trainable | ~10 GB | ~4-6 hrs |

All models use:
- **PaliGemma 2B** vision-language model (LoRA)
- **Gemma 300M** action expert (LoRA)
- Pre-trained base weights from Google Cloud
- Fine-tuned on task-specific demonstrations

## Current Task: Peg Insertion

- **Robot**: Universal Robots UR5e
- **Task**: Insert peg into hole
- **Demonstrations**: 9 episodes
- **Cameras**: 2 (overhead + wrist-mounted)
- **Action space**: 7-DOF (6 joints + gripper)
- **Control**: 30 Hz, delta joint positions

## Quick Start

### Train on HPC

    # SSH to cluster
    ssh saifi@hpc-headnode.iis.fhg.de
    cd /data/beegfs/home/saifi/rlt_ur5e/hpc

    # First time: setup environment (~25 min)
    bash setup_hpc_env.sh
    huggingface-cli login
    wandb login

    # Submit all 3 models
    bash 03_train.sh all

    # Monitor
    squeue -u saifi
    tail -f /data/beegfs/home/saifi/logs/pi0_peg_<JOBID>.out
    # W&B: https://wandb.ai/saifi/openpi

### Download Checkpoints

    # From local machine:
    bash hpc/05_download_checkpoints.sh

### Deploy for Inference

    # On a machine with GPU:
    cd openpi_ur5e/openpi-ur5e
    python scripts/serve_policy.py \
        --config=pi0_ur5e_peg_insertion_lora \
        --checkpoint=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_9demos/30000

## HPC Cluster Details

| Component | Specification |
|-----------|---------------|
| OS | CentOS 7 (glibc 2.17) |
| GPUs | NVIDIA V100 32GB (SXM2) |
| Filesystem | BeeGFS (network) |
| Scheduler | SLURM |
| Solution | sysroot glibc 2.28 + patchelf |

See [hpc/HPC_SETUP_README.md](hpc/HPC_SETUP_README.md) for full technical details.

## Adding New Tasks

See [hpc/ADDING_NEW_DATASET.md](hpc/ADDING_NEW_DATASET.md) for:
- Dataset format (LeRobot v2.0)
- How to create training configs
- Computing normalization statistics
- Hyperparameter guidance

## Experiment Tracking

All runs log to **Weights & Biases**:
- Project: https://wandb.ai/saifi/openpi
- Metrics: loss, learning_rate, grad_norm
- Images: camera views logged at step 0

## References

- [OpenPI Paper](https://arxiv.org/abs/2407.01906) — π0 foundation model
- [Physical Intelligence](https://www.physicalintelligence.company/) — Original authors
- [Original openpi-ur5e](https://github.com/F-Fer/openpi-ur5e) — UR5e adaptation
- [LeRobot](https://github.com/huggingface/lerobot) — Dataset format
