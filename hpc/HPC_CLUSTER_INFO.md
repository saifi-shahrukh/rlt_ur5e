# Fraunhofer IIS HPC Cluster — Full Reference

## Cluster Overview

| Property | Value |
|----------|-------|
| Headnode | hpc-headnode.iis.fhg.de |
| OS | CentOS 7 (glibc 2.17) |
| Scheduler | SLURM |
| Filesystem | BeeGFS (network, no hardlinks) |
| User home | /data/beegfs/home/saifi |
| Internet | Available on headnode + compute nodes |
| SSH | Key-based or password from internal network |

## GPU Partition

| Property | Value |
|----------|-------|
| Partition name | gpu |
| Nodes | 7 (hpc-gpu-02 to hpc-gpu-08) |
| GPUs per node | 4x Tesla V100-SXM2-32GB |
| Total GPUs | 28 |
| Max time | UNLIMITED (no wall-time cap) |
| Default time | NONE (runs indefinitely) |
| Max nodes per job | UNLIMITED |
| Max CPUs per node | UNLIMITED |
| Max memory per node | UNLIMITED |
| Oversubscribe | NO (exclusive GPU allocation) |
| Total CPUs | 392 across 7 nodes (~56 per node) |

## GPU Specifications (Tesla V100-SXM2-32GB)

| Property | Value |
|----------|-------|
| Architecture | Volta (SM 7.0) |
| VRAM | 32 GB HBM2 |
| Memory bandwidth | 900 GB/s |
| FP32 performance | 15.7 TFLOPS |
| FP16 performance | 125 TFLOPS (Tensor Cores) |
| NVLink | Yes (between GPUs on same node) |

## Resource Limits

- NO time limit — jobs run until completion
- NO QOS restrictions — normal QOS with no caps
- NO per-user job limits — submit as many as needed
- 4 GPUs max per node — use --gres=gpu:tesla:4 for full node
- Other users run jobs for 15+ days without issues

## SLURM Commands

    # Check GPU node status
    sinfo -p gpu -o "%n %G %t %C %m" --Node

    # Check partition limits
    sinfo -p gpu -o "%P %l %a %D %G"

    # Detailed partition config
    scontrol show partition gpu

    # See who is using GPUs
    squeue -p gpu -o "%u %j %D %C %b %T %M"

    # Your jobs
    squeue -u saifi

    # Submit a job
    sbatch hpc/slurm/pi0_v2_full.sh

    # Cancel a job
    scancel <JOBID>

    # Watch logs
    tail -f /data/beegfs/home/saifi/logs/<jobname>_<jobid>.err

## Software Stack

| Component | Location |
|-----------|----------|
| Python 3.11 | .venv/bin/python3.11 |
| uv (package manager) | ~/.local/bin/uv |
| micromamba | ~/micromamba/ |
| FFmpeg 7 libs | ~/micromamba/envs/openpi/lib/ |
| cuda-nvcc 12.9 | ~/micromamba/envs/openpi/bin/ |
| OpenPI source | /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/ |
| Virtual env | /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv/ |
| Checkpoints | /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/ |
| SLURM logs | /data/beegfs/home/saifi/logs/ |

## Multi-GPU Training (FSDP)

OpenPI uses JAX FSDP for multi-GPU training. Key settings:

- batch_size must be divisible by jax.device_count()
- fsdp_devices controls model sharding (default=1 = pure data parallel)
- With 4 GPUs: each GPU processes batch_size/4 samples
- Model weights automatically sharded across GPUs for large params

Example SLURM header for 4-GPU training:

    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:tesla:4
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=128G

With batch_size=8 and 4 GPUs: 2 samples per GPU.

## Training Speed Estimates (V100)

| Model | 1 GPU | 4 GPUs (estimated) |
|-------|-------|--------------------|
| pi0 v2 (2-cam) | 6.2s/step | ~1.6s/step |
| pi05 v2 (2-cam) | 7.8s/step | ~2.0s/step |
| pi0-FAST v2 (2-cam) | 8.7s/step | ~2.2s/step |
| pi0 v1 (3-cam) | ~10s/step | ~2.5s/step |
| pi05 v1 (3-cam) | ~11s/step | ~2.8s/step |

## Known Issues & Fixes

1. glibc 2.17 (CentOS 7): Modern Python packages need 2.28+
   - Fix: sysroot + patchelf with --force-rpath (DT_RPATH propagation)
   - Without --force-rpath, get __clock_nanosleep errors

2. BeeGFS no hardlinks: uv sync uses full copy mode

3. HuggingFace gated models: Need HF_TOKEN + license acceptance
   - PaliGemma tokenizer requires HF login

4. FAST tokenizer: Must be cached offline
   - Run cache_fast_tokenizer.sh before training with TRANSFORMERS_OFFLINE=1

5. ptxas not found: XLA flag disables ptxas requirement
   - XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

## Training Configs (V2 — Current)

| Config Name | Model | Cameras | LoRA Rank |
|-------------|-------|---------|----------|
| pi0_ur5e_peg_insertion_v2_lora | pi0 | 2 (overhead+wrist) | VLM=16, Expert=32 |
| pi05_ur5e_peg_insertion_v2_lora | pi0.5 | 2 (overhead+wrist) | VLM=16, Expert=32 |
| pi0_fast_ur5e_peg_insertion_v2_lora | pi0-FAST | 2 (overhead+wrist) | VLM=16 |

## Dataset

| Property | Value |
|----------|-------|
| HuggingFace ID | saifi/ur5e-peg-insertion-50demos-v2 |
| Episodes | 50 |
| Cameras | 2 (overhead Kinect + wrist RealSense) |
| State dim | 7 (6 joints + 1 gripper) |
| Action dim | 7 (6 joint deltas + 1 gripper) |
| Image resolution | 224x224 (resized by SigLIP) |
| Local cache | /data/beegfs/home/saifi/.cache/huggingface/lerobot/ |

## Network Topology (Transfer Path)

    HPC (hpc-headnode.iis.fhg.de)
         |
         | scp/rsync (password auth)
         v
    WSL2 (r10028, 172.20.234.194)
         |
         | scp/rsync (password auth)
         v
    Linux Workstation (robolab-2, 172.22.1.188)

    NOTE: HPC cannot reach Linux workstation directly.
    All transfers go through WSL2 as relay.
