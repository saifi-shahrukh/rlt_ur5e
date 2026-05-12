# Hyperparameter Guide: V100 32GB Training Optimization

## Hardware: Tesla V100-SXM2-32GB

- Memory: 32 GB HBM2
- Bandwidth: 900 GB/s
- Tensor Cores: 640 (FP16/BF16)
- Nodes: 7 (hpc-gpu-02 to hpc-gpu-08), 4 GPUs each
- CPUs per node: 56
- RAM per node: 187 GB
- Storage: BeeGFS (shared network filesystem)

---

## Model Comparison

| Property | pi0 | pi0.5 | pi0-FAST |
|----------|-----|-------|----------|
| Total params | 3.29B | 3.28B | 2.93B |
| Trainable (LoRA) | 468M (14.2%) | 468M (14.3%) | 421M (14.4%) |
| Frozen params memory | 5.25 GiB | 5.47 GiB | 4.67 GiB |
| Trainable params memory | 1.74 GiB | 1.74 GiB | 1.57 GiB |
| Optimizer state | 3.49 GiB | 5.22 GiB | 4.71 GiB |
| Fixed overhead | ~10.5 GiB | ~12.4 GiB | ~11.0 GiB |
| Action head | Diffusion (continuous) | Diffusion (continuous) | FAST tokenizer (discrete) |
| Action decoding | Flow matching | Flow matching | Autoregressive tokens |

---

## Current Settings (Proven Working)

| Setting | pi0 | pi0.5 | pi0-FAST |
|---------|-----|-------|----------|
| batch_size | 8 | 4 | 8 |
| grad_accumulation | 1 | 2 | 1 |
| effective_batch | 8 | 8 | 8 |
| num_workers | 4 | 4 | 4 |
| XLA_MEM_FRACTION | 0.95 | 0.90 | 0.95 |
| Rate (s/step) | 9.3 | 10.1 | 10.6 |
| Total time (5000 steps) | 12.9 hr | 14.0 hr | 14.7 hr |
| Fits in 12hr? | Barely NO | NO | NO |

---

## Optimized Settings (Maximum Speed, Single GPU)

| Setting | pi0 | pi0.5 | pi0-FAST |
|---------|-----|-------|----------|
| batch_size | 8 | 4 | 8 |
| grad_accumulation | 1 | 2 | 1 |
| effective_batch | 8 | 8 | 8 |
| num_workers | 8 | 8 | 8 |
| XLA_MEM_FRACTION | 0.95 | 0.90 | 0.95 |
| Expected rate | ~8.5s | ~9.5s | ~9.5s |
| Total time (5000 steps) | ~11.8 hr | ~13.2 hr | ~13.2 hr |

Key change: num_workers=8 (more parallel data loading, reduces data starvation)

---

## Memory Budget Breakdown (V100 32GB)

Available GPU memory: 32 GiB
Usable (at 0.95 fraction): 30.4 GiB

### pi0 (batch=8):
  Fixed:       10.5 GiB (params + optimizer)
  Activations: ~8.0 GiB (batch=8)
  XLA buffers: ~3.0 GiB
  Total:       ~21.5 GiB -- SAFE (9 GiB headroom)
  
  batch=16:    ~16 GiB activations = ~29.5 GiB -- RISKY
  batch=12:    ~12 GiB activations = ~25.5 GiB -- POSSIBLE but untested

### pi0.5 (batch=4):
  Fixed:       12.4 GiB (params + optimizer -- larger optimizer)
  Activations: ~9.0 GiB (batch=4)
  XLA buffers: ~3.0 GiB
  Total:       ~24.4 GiB -- SAFE (6 GiB headroom)
  
  batch=8:     ~18 GiB activations = ~33.4 GiB -- OOM!
  batch=6:     ~13.5 GiB activations = ~28.9 GiB -- RISKY (rematerialization)

### pi0-FAST (batch=8):
  Fixed:       11.0 GiB (params + optimizer)
  Activations: ~8.5 GiB (batch=8, FAST tokens slightly larger)
  XLA buffers: ~3.0 GiB
  Total:       ~22.5 GiB -- SAFE (8 GiB headroom)
  
  batch=12:    ~12.5 GiB activations = ~26.5 GiB -- POSSIBLE but untested
  batch=16:    ~17 GiB activations = ~31 GiB -- RISKY

---

## Multi-GPU Options (4x V100 per node)

### Option A: Data Parallel (4 separate jobs)
Run 4 independent training jobs (different configs or experiments).
This is what we do now. Maximum GPU utilization.

### Option B: FSDP (model sharding across GPUs)
OpenPI supports: --fsdp-devices=4
This shards the model across 4 GPUs on one node.

  Pros:
  - Can use much larger batch sizes (batch=32 per GPU = effective 128)
  - Faster convergence (larger effective batch)
  - Fits pi0.5 with batch=8 per GPU

  Cons:
  - Communication overhead between GPUs
  - Slightly lower per-step throughput
  - Uses all 4 GPUs for one model

  Expected with FSDP=4:
  | Model | Batch/GPU | Effective Batch | Rate |
  |-------|-----------|-----------------|------|
  | pi0   | 8         | 32              | ~4-5s/step |
  | pi0.5 | 8         | 32              | ~5-6s/step |
  | pi0-FAST | 8      | 32              | ~5-6s/step |

  To enable: add --fsdp-devices=4 and request --gres=gpu:4

### Option C: Gradient Accumulation (simulate larger batch)
Already used for pi0.5. Can also apply to pi0/pi0-FAST:

  | Config | batch | grad_accum | effective | Steps needed |
  |--------|-------|-----------|-----------|-------------|
  | Conservative | 8 | 1 | 8 | 5000 |
  | Medium | 8 | 2 | 16 | 2500 |
  | Aggressive | 8 | 4 | 32 | 1250 |
  | Maximum | 8 | 8 | 64 | 625 |

  Fewer steps needed = faster wall clock time!
  But: each step takes slightly longer due to accumulation overhead.

---

## Recommended Configurations by Scenario

### Scenario 1: Quick experiment (new task, test if it works)
  Steps: 1000-2000
  batch=8, grad_accum=4, effective=32
  Time: ~3-5 hours per model

### Scenario 2: Full training (production quality)
  Steps: 5000
  batch=8, grad_accum=1, effective=8
  Time: ~12-14 hours per model (may need resume)

### Scenario 3: Maximum speed (FSDP, 4 GPUs)
  Steps: 5000
  batch=8 per GPU, FSDP=4, effective=32
  Time: ~7-8 hours per model
  Script change: --gres=gpu:4 --fsdp-devices=4

### Scenario 4: Many tasks (train all 3 models on multiple tasks)
  Use single-GPU jobs, submit all in parallel
  7 nodes x 4 GPUs = 28 simultaneous jobs possible
  Can train 9+ model-task combinations in 12 hours

---

## Training Steps vs Dataset Size

Rule of thumb for LoRA fine-tuning:
  steps = (n_demos * epochs * avg_episode_length) / effective_batch

  | Demos | Avg Length | Epochs | Eff Batch | Steps |
  |-------|-----------|--------|-----------|-------|
  | 9     | 100       | 55     | 8         | ~6200 |
  | 50    | 100       | 10     | 8         | ~6250 |
  | 50    | 100       | 8      | 8         | 5000  |
  | 100   | 100       | 5      | 8         | ~6250 |
  | 200   | 100       | 3      | 8         | ~7500 |
  | 50    | 100       | 5      | 32        | ~780  |

  For 50 demos, 5000 steps with batch=8 gives ~8 epochs.
  This is sufficient for LoRA convergence.

---

## Data Loading Optimization

| Workers | Bottleneck | Notes |
|---------|-----------|-------|
| 1 | Data loading | GPU starved, very slow |
| 2 | Data loading | Still slow |
| 4 | Balanced | Current setting, good |
| 8 | Compute-bound | Optimal for V100 |
| 12 | Diminishing returns | Extra CPU overhead |
| 16 | Wasteful | No benefit, memory pressure |

Recommendation: num_workers=8 with cpus-per-task=12 in SLURM

---

## XLA Compilation Cache

First step is always slow (~40s) due to XLA compilation.
Subsequent steps benefit from cache: ~/.cache/jax/

For resume jobs, the cache from the previous run persists on BeeGFS,
so the first step after resume is also fast.

---

## Summary: Fastest Possible Settings

For a SINGLE V100 32GB, to complete 5000 steps within 12 hours:

  pi0:
    batch=8, workers=8, steps=5000
    Expected: ~8.5s/step = 11.8 hr (tight but possible)
    
  pi0-FAST:
    batch=8, workers=8, steps=5000
    Expected: ~9.5s/step = 13.2 hr (needs resume or fewer steps)
    Alternative: batch=8, grad_accum=2, steps=2500 = ~6.6 hr
    
  pi0.5:
    batch=4, grad_accum=2, workers=8, steps=5000
    Expected: ~9.5s/step = 13.2 hr (needs resume or fewer steps)
    Alternative: batch=4, grad_accum=4, steps=2500 = ~6.6 hr

To guarantee ALL models finish in 12 hours without resume:
    Use grad_accum=2 and steps=3000 for all models.
    Effective batch=16, ~3000 steps = sufficient for LoRA on 50 demos.
