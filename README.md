# RLT UR5e — Robot Learning & Training for UR5e Manipulation

Fine-tune Physical Intelligence's **OpenPI** foundation models (π0, π0.5, π0-FAST)
for UR5e robot manipulation tasks using LoRA on the Fraunhofer IIS HPC cluster.

## Project Structure

    rlt_ur5e/
    ├── README.md                           # This file
    ├── hpc/                                # HPC training scripts & docs
    │   ├── README.md                       # Training guide & quick reference
    │   ├── HPC_SETUP_README.md             # Detailed setup & troubleshooting
    │   ├── ADDING_NEW_DATASET.md           # Guide for new datasets
    │   ├── hpc_gpu.md                      # GPU hardware specs & memory budget
    │   ├── 01_setup.sh                     # Full environment setup
    │   ├── 03_train.sh                     # Submit training jobs (dispatcher)
    │   ├── fix_python_rpath.sh             # Fix DT_RPATH (one-time)
    │   ├── cache_fast_tokenizer.sh         # Cache FAST tokenizer (one-time)
    │   ├── fix_ptxas.sh                    # Install ptxas from nvidia channel
    │   └── slurm/                          # SLURM job scripts
    │       ├── pi0_50demos.sh              # π0 training (50 demos, optimized)
    │       ├── pi05_50demos.sh             # π0.5 training (50 demos, optimized)
    │       ├── pi0_fast_50demos.sh         # π0-FAST training (50 demos, optimized)
    │       ├── pi0.sh                      # π0 training (9 demos, debug)
    │       ├── pi05.sh                     # π0.5 training (9 demos, debug)
    │       └── pi0_fast.sh                 # π0-FAST training (9 demos, debug)
    ├── openpi_ur5e/                        # OpenPI framework
    │   └── openpi-ur5e/
    │       ├── src/openpi/                 # Model & training code
    │       ├── scripts/train.py            # Training entry point
    │       ├── assets/                     # Pre-computed norm stats (all 3 configs)
    │       ├── pyproject.toml              # Package definition
    │       └── uv.lock                     # Pinned dependencies (222 packages)
    └── update_on_readme.md                 # Development notes

## Models

| Model | Architecture | Trainable Params | Fixed Memory | Training Time |
|-------|-------------|------------------|--------------|---------------|
| **π0** | Flow matching + VLM | 468M (14.2%) | 10.48 GiB | ~7-8 hrs |
| **π0.5** | + Cross-attention | 467M (13.7%) | 12.43 GiB | ~10-11 hrs |
| **π0-FAST** | Discrete action tokens | 422M (14.4%) | 10.95 GiB | ~5-6 hrs |

All models use:
- **PaliGemma 2B** vision-language model (LoRA)
- **Gemma 300M** action expert (LoRA)
- Pre-trained base weights from Google Cloud
- Fine-tuned on task-specific demonstrations

## Current Task: Peg Insertion

- **Robot**: Universal Robots UR5e
- **Task**: Insert peg into hole
- **Demonstrations**: 50 episodes (primary), 9 episodes (quick test)
- **Cameras**: 2 (overhead + wrist-mounted)
- **Action space**: 7-DOF (6 joints + gripper)
- **Control**: 30 Hz, delta joint positions
- **Training steps**: 5,000 (LoRA converges quickly)

## Quick Start

### Train on HPC

    # SSH to cluster
    ssh saifi@hpc-headnode.iis.fhg.de
    cd /data/beegfs/home/saifi/rlt_ur5e
    git pull
    cd hpc

    # Submit all 3 models (50 demos, optimized)
    bash 03_train.sh 50demos

    # Or individual models:
    bash 03_train.sh pi0_50
    bash 03_train.sh pi05_50
    bash 03_train.sh pi0fast_50

    # Monitor
    squeue -u saifi
    tail -f /data/beegfs/home/saifi/logs/*_50_*.err
    # W&B: https://wandb.ai/saifi/openpi

### Download Checkpoints

    # From local machine:
    bash hpc/05_download_checkpoints.sh

### Deploy for Inference

    # On a machine with GPU:
    cd openpi_ur5e/openpi-ur5e
    python scripts/serve_policy.py \
        --config=pi0_ur5e_peg_insertion_lora \
        --checkpoint=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/5000

## HPC Cluster Details

| Component | Specification |
|-----------|---------------|
| OS | CentOS 7 (glibc 2.17) |
| GPUs | 28× NVIDIA V100 32GB (SXM2) across 7 nodes |
| Filesystem | BeeGFS (shared network) |
| Scheduler | SLURM |
| Python env | uv + sysroot glibc 2.28 + patchelf (DT_RPATH) |
| Per-job allocation | 1 GPU, 8 CPUs, 64GB RAM |

See [hpc/README.md](hpc/README.md) for training guide and [hpc/HPC_SETUP_README.md](hpc/HPC_SETUP_README.md) for full technical setup.

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
