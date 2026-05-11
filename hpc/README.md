# HPC Training Guide — OpenPI UR5e Peg Insertion

Train π0, π0.5, and π0-FAST models via LoRA fine-tuning on Fraunhofer IIS SLURM cluster.

- **Cluster**: CentOS 7, Tesla V100 32GB, SLURM scheduler
- **Models**: pi0, pi0.5, pi0-FAST (all LoRA fine-tuning)
- **Dataset**: 50 demonstrations of UR5e peg insertion task
- **Monitoring**: https://wandb.ai/saifi/openpi

---

## Quick Start (submit training)

    cd /data/beegfs/home/saifi/rlt_ur5e
    git pull
    cd hpc
    bash 03_train.sh 50demos

## First-Time Setup (one-time only)

See HPC_SETUP_README.md for full installation. Key one-time steps:

    # 1. Install environment (uv sync with sysroot/patchelf)
    bash 01_setup.sh

    # 2. Fix Python RPATH (enables multiprocessing workers)
    bash fix_python_rpath.sh

    # 3. Cache FAST tokenizer (compute nodes have no internet)
    bash cache_fast_tokenizer.sh

    # 4. Login to HuggingFace (PaliGemma is gated)
    huggingface-cli login

    # 5. Prepare dataset (transfer + symlink)
    # Dataset at: ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-50demos-v2
    # Norm stats at: openpi-ur5e/assets/<config>/saifi/ur5e-peg-insertion-dual/

## Monitor Training

    squeue -u saifi                                    # Check job status
    tail -f /data/beegfs/home/saifi/logs/*_50_*.err     # Live logs
    scancel -u saifi                                   # Cancel all jobs

    # W&B dashboard:
    # https://wandb.ai/saifi/openpi

## Submit Individual Models

    bash 03_train.sh pi0_50        # Only π0
    bash 03_train.sh pi05_50       # Only π0.5
    bash 03_train.sh pi0fast_50    # Only π0-FAST
    bash 03_train.sh 50demos       # All 3 (staggered 30s)

---

## Training Configuration

| Model | Config Name | Batch | Grad Accum | Eff. Batch | Workers | Steps | Est. Time |
|-------|-------------|-------|------------|------------|---------|-------|-----------|
| **π0** | pi0_ur5e_peg_insertion_lora | 8 | 1 | 8 | 4 | 5000 | ~7-8 hrs |
| **π0.5** | pi05_ur5e_peg_insertion_lora | 4 | 2 | 8 | 4 | 5000 | ~10-11 hrs |
| **π0-FAST** | pi0_fast_ur5e_peg_insertion_lora | 8 | 1 | 8 | 4 | 5000 | ~5-6 hrs |

### Why 5000 steps?

- LoRA fine-tuning converges quickly (only ~14% params trainable)
- 50 demos × 5000 steps ÷ 8 batch = 625 epochs through data
- Sufficient for task-specific adaptation with pretrained base
- Fits within 12-hour SLURM time limit

---

## Memory Budget (V100 32GB)

| Component | π0 | π0.5 | π0-FAST |
|-----------|-----|-------|----------|
| Frozen params | 5.25 GiB | 5.47 GiB | 4.67 GiB |
| Trainable (LoRA) | 1.74 GiB | 1.74 GiB | 1.57 GiB |
| Optimizer (Adam) | 3.49 GiB | 5.22 GiB | 4.71 GiB |
| **Fixed total** | **10.48 GiB** | **12.43 GiB** | **10.95 GiB** |
| Activations (batch=8) | ~8 GiB ✓ | ~19 GiB ✗ | ~7 GiB ✓ |
| Activations (batch=4) | ~4 GiB ✓ | ~9 GiB ✓ | ~4 GiB ✓ |

**π0.5 requires batch=4** — batch=8 triggers XLA rematerialization.

---

## Key Technical Challenges & Solutions

### 1. glibc Version Mismatch

**Problem**: CentOS 7 has glibc 2.17; JAX/PyTorch need glibc ≥ 2.28.

**Solution**: conda sysroot + patchelf:
- Sysroot installed at .venv/x86_64-conda-linux-gnu/sysroot/ (contains glibc 2.28)
- Python binary patched: interpreter set to sysroot ld-linux-x86-64.so.2
- DT_RPATH set via --force-rpath to sysroot lib paths
- DT_RPATH (not DT_RUNPATH) propagates transitively to ALL loaded .so files

### 2. ptxas/nvlink (CUDA Compiler)

**Problem**: ptxas needs clean system env; if it inherits sysroot libs, it crashes.

**Solution**:
- Do NOT use LD_LIBRARY_PATH (would pollute child processes)
- DT_RPATH only affects the python process (does not propagate via exec())
- ptxas spawned as separate process → uses system glibc 2.17 → works
- Fallback: XLA_FLAGS=--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true

### 3. Multiprocessing Workers

**Problem**: Workers spawn new python3.11 processes that need sysroot libs.

**Solution**: DT_RPATH in python3.11 binary. Workers run same binary → inherit RPATH → find sysroot libs without LD_LIBRARY_PATH.

### 4. Offline Compute Nodes

**Problem**: Compute nodes have no internet. HuggingFace models fail to download.

**Solution**:
- Pre-download all models on headnode (has internet)
- Set HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1
- FAST tokenizer: bash cache_fast_tokenizer.sh (pre-caches on headnode)
- Base checkpoints auto-download from GCS (compute nodes CAN access GCS)

### 5. FFmpeg for torchcodec

**Problem**: torchcodec needs FFmpeg 4-7 (not available on CentOS 7).

**Solution**: micromamba install ffmpeg (in .venv, provides libavutil.so.59).

---

## File Structure

    hpc/
    ├── README.md                    # This file
    ├── HPC_SETUP_README.md          # Detailed installation guide
    ├── 01_setup.sh                  # Full environment setup
    ├── 03_train.sh                  # Job submission dispatcher
    ├── fix_python_rpath.sh          # Fix DT_RPATH (one-time)
    ├── cache_fast_tokenizer.sh      # Cache FAST tokenizer (one-time)
    ├── fix_ptxas.sh                 # Install ptxas from nvidia channel
    └── slurm/
        ├── pi0_50demos.sh           # π0 training (50 demos)
        ├── pi05_50demos.sh          # π0.5 training (50 demos)
        ├── pi0_fast_50demos.sh      # π0-FAST training (50 demos)
        ├── pi0.sh                   # π0 training (9 demos)
        ├── pi05.sh                  # π0.5 training (9 demos)
        └── pi0_fast.sh              # π0-FAST training (9 demos)

---

## Outputs

| Output | Location |
|--------|----------|
| Checkpoints | openpi-ur5e/checkpoints/CONFIG/peg_insertion_50demos/ |
| Logs (stdout) | /data/beegfs/home/saifi/logs/MODEL_50_JOBID.out |
| Logs (stderr) | /data/beegfs/home/saifi/logs/MODEL_50_JOBID.err |
| W&B runs | https://wandb.ai/saifi/openpi |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| __clock_nanosleep / librt error | Run bash fix_python_rpath.sh |
| Can't connect to huggingface.co | Ensure *_OFFLINE=1 flags set; pre-cache on headnode |
| Couldn't find ptxas | Run micromamba install cuda-nvcc -c nvidia |
| rematerialization warning (π0.5) | Reduce batch size to 4 |
| LEROBOT_HOME deprecated | Use HF_LEROBOT_HOME instead |
| Job crashes immediately | Check .err log: cat /data/beegfs/home/saifi/logs/MODEL_50_JOBID.err |
