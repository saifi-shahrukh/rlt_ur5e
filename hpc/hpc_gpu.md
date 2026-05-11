# HPC GPU Cluster — Hardware Specification

## Cluster Overview

| Property | Value |
|----------|-------|
| **Nodes (GPU partition)** | 7 nodes: hpc-gpu-02 through hpc-gpu-08 |
| **GPUs per Node** | 4× NVIDIA Tesla V100 32GB (SXM2) |
| **Total GPUs** | 28× V100 32GB |
| **CPU per Node** | 2× Intel Xeon (14 cores/socket) = 56 threads (HT enabled) |
| **RAM per Node** | 187 GB DDR4 |
| **OS** | CentOS 7 (kernel 3.10.0-1160, glibc 2.17) |
| **Interconnect** | InfiniBand (BeeGFS shared filesystem) |
| **Scheduler** | SLURM 20.11.9 |

## GPU Specifications — Tesla V100 32GB (SXM2)

| Property | Value |
|----------|-------|
| **Architecture** | Volta (SM 7.0, compute capability 7.0) |
| **VRAM** | 32 GB HBM2 |
| **Memory Bandwidth** | 900 GB/s |
| **FP32 Performance** | 15.7 TFLOPS |
| **FP16 Tensor Cores** | 125 TFLOPS |
| **CUDA Cores** | 5120 |
| **Tensor Cores** | 640 (1st gen) |
| **TDP** | 300W |
| **NVLink** | 300 GB/s (within node, 6 links) |

## Node Details

| Node | Partition | GPUs | Notes |
|------|-----------|------|-------|
| hpc-gpu-01 | gpu-short | 4× V100 32GB | Short jobs only |
| hpc-gpu-02 | gpu | 4× V100 32GB | Standard |
| hpc-gpu-03 | gpu | 4× V100 32GB | Standard |
| hpc-gpu-04 | gpu | 4× V100 32GB | Standard |
| hpc-gpu-05 | gpu | 4× V100 32GB | Standard |
| hpc-gpu-06 | gpu | 4× V100 32GB | Standard |
| hpc-gpu-07 | gpu | 4× V100 32GB | Standard |
| hpc-gpu-08 | gpu | 4× V100 32GB | Standard |

## Per-Node Resources

- **CPUs:** 56 total (28 physical cores × 2 HT)
- **RAM:** 187 GB (171 GB usable after OS reservation)
- **GPUs:** 4× V100 32GB with NVLink interconnect
- **Optimal per-job allocation:** 1 GPU + 8-12 CPUs + 48-64 GB RAM
- **Max concurrent jobs per node:** 3-4 (one per GPU)

## Memory Budget for π0 Training (per V100 32GB)

| Component | pi0 | pi0.5 | pi0-FAST |
|-----------|-----|-------|----------|
| Params (frozen) | 5.25 GiB | 5.47 GiB | ~5.2 GiB |
| Params (trainable/LoRA) | 1.74 GiB | 1.74 GiB | ~1.7 GiB |
| Optimizer state (Adam) | 3.49 GiB | 3.48 GiB | ~3.4 GiB |
| **Subtotal (fixed)** | **10.48 GiB** | **10.69 GiB** | **~10.3 GiB** |
| Available for activations | ~21.5 GiB | ~21.3 GiB | ~21.7 GiB |
| Activations (batch=16) | ~14 GiB | ~19 GiB (!) | ~12 GiB |
| Activations (batch=8) | ~8 GiB | ~11 GiB ✓ | ~7 GiB |

**Critical:** pi0.5 with batch=16 triggers XLA rematerialization (needs 18.86 GiB
activations, exceeds safe margin). Use batch=8 + grad_accumulation=2 for pi0.5.

## Expected Training Performance (properly configured)

| Model | Batch | s/step (expected) | 15k steps | 30k steps |
|-------|-------|-------------------|-----------|----------|
| pi0 | 16 | ~1.0-1.5s | 4-6 hrs | 8-12 hrs |
| pi0.5 | 8+accum | ~2.5-3.5s | 10-14 hrs | 20-28 hrs |
| pi0-FAST | 16 | ~0.5-0.8s | 2-3 hrs | 4-7 hrs |

## Key Environment Constraint: CentOS 7 + glibc

**Problem:** CentOS 7 ships glibc 2.17. Modern Python packages (JAX, PyTorch, etc.)
require glibc ≥ 2.28. Solved via conda                        .

**Architecture:**
-                                         — contains glibc 2.28 + ld-linux
- Main Python process launched via:                                                      
- This provides sysroot libs to the main process only (process-local)
- ptxas/nvlink (CUDA compiler tools) need CLEAN system environment

**Critical limitation for data workers:**
-                  is process-local (not inherited by child processes)
- Python multiprocessing         creates NEW processes that don't inherit                 
- Spawned workers fall back to system glibc 2.17 → crash on glibc 2.28 symbols
- This is why                   was used as workaround (no child processes)
- **Fix:** patchelf Python binary interpreter to sysroot ld-linux, OR set LD_LIBRARY_PATH + patchelf ptxas
