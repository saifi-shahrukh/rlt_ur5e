# 🖥️ Local GPU Training — RTX 5070 Ti (16GB)

> **TESTED & WORKING** ✅ — 2.1 it/s, ~3h 53min for 30k steps, 96% VRAM utilization

---

## Quick Start

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

# ⚠️  FIRST: Kill any zombie python processes holding GPU memory!
nvidia-smi
# If you see python processes using VRAM → kill -9 <PID>

# Train π0-FAST LoRA (the only config that fits 16GB)
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_run1 --overwrite
```

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5070 Ti — **16 GB VRAM** |
| CPU | AMD Ryzen 9 9950X 16-Core |
| RAM | 92 GB DDR5 |
| OS | Ubuntu (x86_64) |

---

## What Fits on 16GB?

| Config | Model | VRAM | Fits? | Batch | Rank | Speed |
|--------|-------|------|-------|-------|------|-------|
| **`pi0_fast_ur5e_peg_insertion_lora`** | **π0-FAST LoRA** | **15.7 GB** | **✅ YES** | **1** | **4** | **2.1 it/s** |
| `pi0_ur5e_peg_insertion_lora` | π0 LoRA | ~24 GB | ❌ OOM | — | — | — |
| `pi05_ur5e_peg_insertion_lora` | π0.5 LoRA | ~28 GB | ❌ OOM | — | — | — |

> **Only `pi0_fast_ur5e_peg_insertion_lora` works locally.** The others require the HPC cluster.

---

## Working Configuration (Verified)

```
Model:          π0-FAST (PaLI-Gemma 2B LoRA + FAST action tokenizer)
LoRA rank:      4  (reduced from default 16 → saves ~3.7 GiB)
Batch size:     1  (minimum viable)
Action horizon: 30 steps
Max token len:  180
Training steps: 30,000
Save interval:  Every 5,000 steps

Memory breakdown:
  Frozen params (bf16):   4.67 GiB
  Trainable params (f32): 1.57 GiB  (421M LoRA params)
  Optimizer state:        3.14 GiB  (Adam momentum + variance)
  Forward/backward pass:  ~6.2 GiB
  ─────────────────────────────────
  Total:                  ~15.6 GiB / 16.0 GiB available
```

---

## Environment Variables (Critical!)

The `scripts/train_local.sh` wrapper sets these automatically:

```bash
# Pre-allocate full memory pool (avoids fragmentation → OOM during transfers)
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# Use 95% of GPU memory (default is 75% → wastes 4GB!)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Disable XLA autotuning (saves ~700MB during compilation)
export XLA_FLAGS="--xla_gpu_autotune_level=0"
```

---

## Training Commands

### Fresh Training
```bash
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora \
    --exp-name=peg_insertion_run1 \
    --overwrite
```

### Resume from Checkpoint
```bash
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora \
    --exp-name=peg_insertion_run1 \
    --resume
```

### Run in Background (recommended for long training)
```bash
nohup ./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora \
    --exp-name=peg_insertion_run1 --overwrite \
    > /tmp/train_peg_insertion.log 2>&1 &

# Monitor:
tail -f /tmp/train_peg_insertion.log
```

---

## Monitoring

### Terminal
```bash
# Watch training progress
tail -f /tmp/train_peg_insertion.log

# Check GPU utilization
watch -n 2 nvidia-smi

# Check checkpoints
ls checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/
```

### WandB Dashboard
- **Project:** https://wandb.ai/saifi/openpi
- Tracks: loss, grad_norm, learning_rate, training images

---

## Checkpoints

Saved at:
```
checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/
├── 5000/     ← kept permanently
├── 10000/    ← kept permanently
├── 15000/    ← kept permanently
├── 20000/    ← kept permanently
├── 25000/    ← kept permanently
└── 30000/    ← final checkpoint
```

---

## After Training: Serve & Deploy

### Terminal 1: Serve model
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/30000
```

### Terminal 2: Run on robot
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --fps=30
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during init | `nvidia-smi` → find zombie python processes → `kill -9 <PID>` |
| OOM during training | Should not happen with this config. Check nothing else uses GPU. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` ignored | Must set BEFORE python starts (wrapper does this) |
| Training too slow | 2.1 it/s is expected for batch=1 on 16GB GPU |
| Loss not decreasing | Normal for first ~100 steps. Should drop from ~9 to ~4 within 500 steps. |
| `--resume` error | Use `--resume` (bare flag, NOT `--resume=True`) |
| Checkpoint dir exists | Use `--overwrite` to clear, or `--resume` to continue |

### Loss Trajectory (Expected)
```
Step 0:    loss ≈ 9.4 (random)
Step 100:  loss ≈ 8.5
Step 200:  loss ≈ 5.8
Step 300:  loss ≈ 4.7
Step 400:  loss ≈ 4.3
Step 5000: loss ≈ 2.5 (converging)
```

---

## Why These Settings?

### LoRA rank=4 (instead of default 16)
- Default rank=16: 442M trainable params → 1.65 GiB + 3.3 GiB optimizer = 4.95 GiB
- Rank=4: 421M trainable params → 1.57 GiB + 3.14 GiB optimizer = 4.71 GiB
- **Plus:** Smaller LoRA matrices = less memory during forward/backward pass
- Trade-off: Slightly less model capacity, but fine for few-shot tasks

### Batch size=1 (instead of 8-16)
- Each sample has 3 images × 224×224 × ViT = 768 image tokens
- Forward + backward memory scales linearly with batch size
- Batch=1 is minimum viable — gradient is noisier but model still learns

### XLA memory flags
- `PREALLOCATE=true`: Creates contiguous memory pool → avoids fragmentation OOM
- `MEM_FRACTION=0.95`: Default 0.75 wastes 4GB we desperately need
- `autotune_level=0`: Disables GEMM autotuning that allocates temporary 700MB buffers
