# HPC GPU Cluster — Hardware & Training Configuration

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

## Memory Budget for π0 Training (per V100 32GB)

| Component | π0 | π0.5 | π0-FAST |
|-----------|-----|-------|----------|
| Params (frozen) | 5.25 GiB | 5.47 GiB | ~5.2 GiB |
| Params (trainable/LoRA) | 1.74 GiB | 1.74 GiB | ~1.7 GiB |
| Optimizer state (Adam) | 3.49 GiB | 5.22 GiB | ~3.4 GiB |
| **Subtotal (fixed)** | **10.48 GiB** | **12.43 GiB** | **~10.3 GiB** |
| Available for activations | ~21.5 GiB | ~19.6 GiB | ~21.7 GiB |
| Activations (batch=8) | ~8 GiB ✓ | ~11 GiB ✓ | ~7 GiB ✓ |
| Activations (batch=4) | ~4 GiB ✓ | ~6 GiB ✓ | ~4 GiB ✓ |

**Critical:** π0.5 with batch≥8 triggers XLA rematerialization (21.77 GiB needed,
only 16.82 GiB safe after overhead). Use batch=4 + grad_accum=2.

## Optimal Training Configuration (V100 32GB)

| Model | Batch | Grad Accum | Eff. Batch | Workers | Steps | Est. Time |
|-------|-------|------------|------------|---------|-------|----------|
| **π0** | 8 | 1 | 8 | 4 | 5000 | ~7-8 hrs |
| **π0.5** | 4 | 2 | 8 | 4 | 5000 | ~10-11 hrs |
| **π0-FAST** | 8 | 1 | 8 | 4 | 5000 | ~5-6 hrs |

### Why 5000 steps?
- LoRA fine-tuning converges quickly (only 14% params trainable)
- 50 demos × 5000 steps ÷ 8 batch = 625 full passes through data
- Official openpi uses 30k steps with 32 batch on 8× H100 → equivalent data seen
- We process same #samples: 5000×8 = 40k vs 30k×32÷8GPUs = 120k (conservative)

## Environment Architecture

**Problem:** CentOS 7 has glibc 2.17. JAX/PyTorch need glibc ≥ 2.28.

**Solution:** patchelf + sysroot (DT_RPATH):
1.              binary has ELF interpreter → sysroot                       
2.              binary has DT_RPATH →                                            
3. DT_RPATH (unlike DT_RUNPATH) propagates **transitively** to all loaded       files
4. This means jaxlib, torch, etc. all find sysroot glibc versions
5. ptxas (spawned as separate process) is NOT affected by parent's RPATH

**One-time setup:**                               

## Quick Start

``                                                                                                                                                                                                                                                                                                    ``
