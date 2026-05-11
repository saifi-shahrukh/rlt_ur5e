# HPC Training Scripts

Train π0, π0.5, and π0-FAST models on Fraunhofer IIS HPC cluster.

## What This Does

Fine-tunes Physical Intelligence's **OpenPI** foundation models (π0 family) using
**LoRA** on 9 demonstrations of a UR5e robot performing peg insertion with dual cameras.

## Models

| Model | Description | Training Time | VRAM |
|-------|-------------|---------------|------|
| **π0** | Standard flow matching policy | ~8-10 hrs | ~16 GB |
| **π0.5** | Enhanced cross-attention architecture | ~10-12 hrs | ~28 GB |
| **π0-FAST** | Lightweight continuous action tokens | ~4-6 hrs | ~10 GB |

## Usage

    # One-time setup (~25 min)
    bash setup_hpc_env.sh

    # Login to HuggingFace + W&B
    huggingface-cli login
    wandb login

    # Train (choose: pi0, pi05, pi0_fast, both, all)
    bash 03_train.sh all

    # Monitor
    squeue -u saifi
    tail -f /data/beegfs/home/saifi/logs/pi0_peg_<JOBID>.out
    # W&B: https://wandb.ai/saifi/openpi

    # Download checkpoints to local machine
    bash 05_download_checkpoints.sh

## Dataset

**UR5e Peg Insertion (Dual Camera)**
- 9 expert demonstrations
- 2 cameras: overhead + wrist-mounted
- 7-DOF: 6 joint positions + 1 gripper
- 30 Hz control frequency
- Location: /data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual/

## Scripts

| Script | Purpose |
|--------|─────────|
| setup_hpc_env.sh | Full environment setup (glibc 2.28 + packages) |
| 03_train.sh | Submit SLURM training jobs |
| 04_status.sh | Check job status + recent logs |
| 05_download_checkpoints.sh | SCP checkpoints to local |
| 06_interactive.sh | Interactive GPU session for debugging |

## Detailed Documentation

See [HPC_SETUP_README.md](HPC_SETUP_README.md) for:
- Full setup walkthrough
- Architecture diagram
- Troubleshooting guide
- Design decisions
