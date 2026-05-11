# HPC Setup Guide — OpenPI Training on Fraunhofer IIS Cluster

## Overview

Train **π0**, **π0.5**, and **π0-FAST** LoRA models for UR5e peg insertion on the
Fraunhofer IIS HPC cluster (CentOS 7, NVIDIA V100 32GB GPUs, SLURM scheduler).

**Core challenge:** CentOS 7 ships glibc 2.17, but modern ML packages (JAX 0.5.3,
PyTorch 2.7.1, tensorstore, etc.) require glibc >= 2.28. We solve this with a
**sysroot + patchelf + ld-linux** approach — no root access needed.

---

## Quick Start (If Already Setup)

    # SSH to HPC
    ssh saifi@hpc-headnode.iis.fhg.de

    # Submit training (pi0, pi05, pi0_fast, both, or all)
    cd /data/beegfs/home/saifi/rlt_ur5e/hpc
    bash 03_train.sh all

    # Monitor
    squeue -u saifi
    tail -f /data/beegfs/home/saifi/logs/pi0_peg_<JOBID>.out

    # W&B dashboard
    # https://wandb.ai/saifi/openpi

---

## Architecture

    +------------------+     +-------------------+     +------------------+
    |   Local (WSL2)   | --> |   HPC Headnode    | --> |   GPU Nodes      |
    |   git push       |     |   sbatch submit   |     |   V100 32GB      |
    +------------------+     +-------------------+     +------------------+

    Environment Stack (on GPU nodes):
    ┌─────────────────────────────────────────────────────────────────┐
    │  SLURM Job                                                      │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │  ld-linux-x86-64.so.2 (from sysroot, glibc 2.28)         │  │
    │  │  ┌─────────────────────────────────────────────────────┐  │  │
    │  │  │  Python 3.11.15 (.venv/bin/python3.11)              │  │  │
    │  │  │  ┌───────────────────────────────────────────────┐  │  │  │
    │  │  │  │  JAX 0.5.3 + PyTorch 2.7.1+cu126 + OpenPI    │  │  │  │
    │  │  │  │  All .so files loaded via glibc 2.28 loader   │  │  │  │
    │  │  │  └───────────────────────────────────────────────┘  │  │  │
    │  │  └─────────────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

---

## Full Setup (From Scratch)

### Prerequisites

1. SSH access to                          
2. Dataset at                                                                  
3. HuggingFace account with PaliGemma access (gated model)
4. W&B account for experiment tracking

### Step 1: Clone Repository

    ssh saifi@hpc-headnode.iis.fhg.de
    cd /data/beegfs/home/saifi
    git clone https://github.com/saifi-shahrukh/rlt_ur5e.git
    cd rlt_ur5e/hpc

### Step 2: Run Setup Script

    bash setup_hpc_env.sh

This script:
- Installs micromamba (if not present)
- Creates         with Python 3.11 + sysroot_linux-64=2.28 + patchelf
- Patches Python binary to use glibc 2.28 dynamic linker
- Runs           to install all 222 packages from          
- Creates dataset symlink for lerobot
- Verifies all imports work

Takes ~25 minutes (mostly downloading wheels + copying due to BeeGFS).

### Step 3: Login to Services

    # HuggingFace (required for PaliGemma tokenizer)
    # First: accept license at https://huggingface.co/google/paligemma-3b-pt-224
    huggingface-cli login

    # W&B (for experiment tracking)
    wandb login

### Step 4: Submit Training

    bash 03_train.sh all    # submits pi0, pi0.5, pi0-FAST (staggered)

---

## Models

| Model | Config Name | Base Weights | VRAM | Batch | Notes |
|-------|-------------|--------------|------|-------|-------|
| **π0** |                               |                                                  | ~14-16 GB | 4 | Standard flow matching |
| **π0.5** |                                |                                                   | ~24-32 GB | 16 | Cross-attention VLM+expert |
| **π0-FAST** |                                    |                                                       | ~8-12 GB | 16 | Lightweight, fastest |

All use:
- **LoRA** fine-tuning:                 (VLM) +                   (action expert)
- **30,000 training steps** with checkpoint every 5,000
- **9 demonstrations** of UR5e peg insertion with dual cameras
- **Delta actions** (relative joint positions)

---

## Dataset

    /data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual/
    ├── data/               # Parquet files with actions & states
    ├── meta/               # Dataset metadata (episodes, tasks, info)
    └── videos/             # MP4 recordings (overview_cam, wrist_cam)

- **9 episodes** of peg insertion demonstrations
- **2 cameras**: overhead (overview_cam) + wrist-mounted (wrist_cam)
- **7-DOF actions**: 6 joint positions + 1 gripper
- **30 Hz** control frequency
- Symlinked to                                                             

---

## Monitoring

### Terminal

    squeue -u saifi                                    # job status
    tail -f /data/beegfs/home/saifi/logs/pi0_peg_<ID>.out   # stdout
    cat /data/beegfs/home/saifi/logs/pi0_peg_<ID>.err       # errors
    scancel <JOBID>                                    # kill job

### W&B Dashboard

    https://wandb.ai/saifi/openpi

The train.py hardcodes                          . Metrics logged:
-        — training loss (should decrease)
-                 — cosine schedule
-             — gradient magnitude
-                — sample images at step 0

---

## Checkpoints

Saved to:

    openpi_ur5e/openpi-ur5e/checkpoints/<config>/<exp_name>/
    ├── 5000/          # step 5000 (kept permanently)
    ├── 10000/
    ├── 15000/
    ├── 20000/
    ├── 25000/
    ├── 30000/         # final
    └── latest -> 30000/

Each checkpoint contains:
-           — model weights (for inference)
-                — optimizer state (for resuming)
-           — norm stats and config

### Download Checkpoints to Local

    bash 05_download_checkpoints.sh

---

## Troubleshooting

### "ml_dtypes has no attribute float8_e3m4"

**Cause:**           ran without          , resolving tensorstore==0.1.83 (needs ml-dtypes>=0.5.0)
but override forces ml-dtypes==0.4.1.

**Fix:** Ensure                                   exists (pins tensorstore==0.1.74).
Re-run:                                        

### "munmap_chunk(): invalid pointer" / tensorstore crash

**Cause:** Mixing glibc versions — LD_LIBRARY_PATH sets sysroot libs but system's
ld-linux (glibc 2.17) is still the loader.

**Fix:** Use sysroot's ld-linux directly (already in SLURM scripts):

    ${SYSROOT}/lib64/ld-linux-x86-64.so.2 --library-path ${SYSROOT}/lib64:... python3.11

NEVER use                   with sysroot paths globally.

### "Repository Not Found" for dataset

**Cause:**                          queries HuggingFace Hub, but dataset is local-only.

**Fix:** Set                    and                         (already in SLURM scripts).
Also clean broken cache:                                                                      

### "Failed to download PaliGemma tokenizer"

**Cause:** PaliGemma is a gated model requiring license acceptance.

**Fix:**
1. Accept at https://huggingface.co/google/paligemma-3b-pt-224
2. Run                         on HPC

### Job dies in <30 seconds

**Check:**                                                    

Common causes:
- Missing HF_TOKEN (see above)
- Broken dataset cache (clean .incomplete dirs)
- Wrong Python invocation (should use ld-linux, not bare python)

### Jobs conflict on BeeGFS (race condition)

**Cause:** Multiple jobs creating dataset cache simultaneously on network filesystem.

**Fix:**               now staggers submissions by 60 seconds.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| sysroot_linux-64=2.28 | Provides glibc 2.28 headers and libs without root |
| patchelf on Python | Changes ELF interpreter so Python loads sysroot's ld-linux |
|                           | All .so loading goes through glibc 2.28 — no version mixing |
|           (not pip) | Reads pre-resolved uv.lock — no backtracking, 40s resolution |
|           committed | Pins exact versions (tensorstore==0.1.74 compatible with ml-dtypes==0.4.1) |
|                    | Dataset is local; prevents 404 from Hub |
| Staggered submissions | Avoids BeeGFS cache race condition |
|                    | Permanent checkpoints every 5k steps |

---

## File Structure

    hpc/
    ├── 01_setup.sh                 # Initial setup wrapper
    ├── 02_transfer_dataset.sh      # Transfer dataset to HPC
    ├── 03_train.sh                 # Submit training jobs
    ├── 04_status.sh                # Check job status
    ├── 05_download_checkpoints.sh  # Download results
    ├── 06_interactive.sh           # Interactive GPU session
    ├── setup_hpc_env.sh            # Full environment setup
    ├── HPC_SETUP_README.md         # This file
    ├── README.md                   # Brief overview
    └── slurm/
        ├── pi0.sh                  # SLURM script for π0
        ├── pi05.sh                 # SLURM script for π0.5
        └── pi0_fast.sh             # SLURM script for π0-FAST
