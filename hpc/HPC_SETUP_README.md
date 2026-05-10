# HPC Environment Setup — Definitive Guide

> **Cluster:** Fraunhofer IIS HPC (CentOS 7, V100 32GB, SLURM)  
> **Problem:** CentOS 7 has glibc 2.17, but modern ML packages need glibc 2.28+  
> **Solution:** conda sysroot + patchelf (NO version hacks, NO --no-deps)

---

## TL;DR — Run This

```bash
# SSH to HPC headnode
ssh hpc-headnode.iis.fhg.de

# One-time setup (creates env, patches Python, installs everything)
cd /data/beegfs/home/saifi/rlt_ur5e/hpc
bash setup_hpc_env.sh

# Login to W&B
source /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv/activate_hpc.sh
wandb login

# Submit training
bash 03_train.sh both
```

---

## Why Previous Approaches Failed

### Approach 1: `01_setup.sh` (micromamba + `pip install -e .`)

**What it did:** Installed Python 3.11 + CUDA via conda, then ran `pip install -e .`

**Why it failed:** conda's Python was still linked to the system's glibc 2.17. When pip
tried to install PyTorch 2.7.1 (manylinux_2_28), pip checked glibc, saw 2.17, and refused
to use the wheel. On some systems it falls back to building from source (impossible
without gcc ≥ 9).

### Approach 2: `install_env.sh` (downgraded versions + `--no-deps`)

**What it did:** Pinned old versions (torch 2.5.1, jax 0.4.30) that have manylinux_2_17
wheels. Used `--no-deps` on packages whose deps needed glibc 2.28.

**Why it failed:** The cascading consequences of `--no-deps`:
- `lerobot --no-deps` → missing `datasets`, `accelerate`, `av`, `gymnasium`
- `datasets --no-deps` → missing `pyarrow-hotfix`, `dill`, `multiprocess`, `xxhash`
- `flax==0.10.2` → incompatible with `jax 0.4.30` (needs ≥ 0.4.34)
- `orbax-checkpoint==0.11.13` → incompatible with `jax 0.4.30` (needs ≥ 0.5)

Result: An impossible dependency maze where fixing one thing breaks another.

---

## How The New Setup Works

### The Key Insight: sysroot + patchelf

```
┌─────────────────────────────────────────────────────────────┐
│  System (CentOS 7)         │  Conda Environment            │
│  glibc 2.17 ─ UNTOUCHED   │  sysroot_linux-64 = 2.28      │
│  /lib64/ld-linux...        │  ${VENV}/x86_64-conda-linux-  │
│                            │    gnu/sysroot/lib64/          │
│                            │      ├── libc.so.6 (2.28)     │
│                            │      └── ld-linux-x86-64.so.2 │
└────────────────────────────┴────────────────────────────────┘

patchelf changes Python's ELF interpreter:
  BEFORE: /lib64/ld-linux-x86-64.so.2       (→ loads glibc 2.17)
  AFTER:  ${VENV}/.../sysroot/.../ld-linux-x86-64.so.2 (→ loads glibc 2.28)

Result: pip sees glibc 2.28 → downloads manylinux_2_28 wheels → everything works!
```

### What Gets Installed

Exact versions from `pyproject.toml` (no downgrades!):

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | From conda |
| torch | 2.7.1 | cu124 wheel |
| jax[cuda12] | 0.5.3 | Official version |
| flax | 0.10.2 | Compatible with jax 0.5.3 |
| orbax-checkpoint | 0.11.13 | Compatible with jax 0.5.3 |
| lerobot | ≥0.4.0 | Full install (no --no-deps) |
| transformers | 4.53.2 | Full install |
| numpy | ≥1.22.4 | Whatever pip resolves |
| polars | ≥1.30.0 | Full install |
| All others | As pinned | Full dependency tree |

### Only Exception: rerun-sdk

`lerobot` optionally depends on `rerun-sdk` (3D visualization). This needs X11/OpenGL
which is unavailable on headless HPC nodes. The setup script handles this automatically:
- Tries full install first
- If rerun-sdk fails, installs lerobot with `--no-deps` but manually adds ALL of lerobot's
  other runtime dependencies (datasets, pyarrow, etc.)

This is the **only** `--no-deps` usage, and it's tightly controlled.

---

## File Layout

```
hpc/
├── setup_hpc_env.sh          ← ONE-TIME SETUP (run this!)
├── 03_train.sh               ← Submit training jobs
├── 04_status.sh              ← Check job status
├── 05_download_checkpoints.sh← Pull checkpoints to local
├── 06_interactive.sh         ← Interactive GPU debug session
├── slurm/
│   ├── pi0.sh                ← SLURM job: π0 LoRA training
│   └── pi05.sh               ← SLURM job: π0.5 LoRA training
├── HPC_SETUP_README.md       ← This file
│
├── 01_setup.sh               ← [DEPRECATED] Old micromamba approach
├── install_env.sh            ← [DEPRECATED] Old --no-deps approach
├── fix_and_run.sh            ← [DEPRECATED] Patch attempts
└── fix_all_deps.sh           ← [DEPRECATED] Patch attempts
```

---

## SLURM Jobs

Both SLURM scripts (`slurm/pi0.sh`, `slurm/pi05.sh`) source the activation script:

```bash
source "${VENV}/activate_hpc.sh"
```

This sets:
- `PATH` → conda Python first
- `LD_LIBRARY_PATH` → sysroot glibc 2.28 libs first
- `HF_HOME` → BeeGFS cache
- `XLA_PYTHON_CLIENT_MEM_FRACTION` → 90% GPU memory for JAX
- `WANDB_PROJECT` → rlt-ur5e

---

## Troubleshooting

### "GLIBC_2.28 not found" at runtime

```bash
source /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv/activate_hpc.sh
ldd $(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib/libtorch.so'))")
```

If you see "not found" for glibc symbols, the `LD_LIBRARY_PATH` isn't set. Source the
activation script.

### "No module named X"

The full install should have gotten everything. But if something is missing:
```bash
source .venv/activate_hpc.sh
pip install <missing-package>
```

Since glibc 2.28 is working, pip will download the correct wheel.

### GPU not found / CUDA error

Check the GPU driver version on compute nodes:
```bash
srun --partition=gpu --gres=gpu:1 nvidia-smi
```

Needs driver ≥ 525 for CUDA 12.4. If older, change torch index to `cu118`.

### Rebuilding from scratch

```bash
rm -rf /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv
cd /data/beegfs/home/saifi/rlt_ur5e/hpc
bash setup_hpc_env.sh
```

---

## W&B Setup

```bash
source .venv/activate_hpc.sh
wandb login
# Paste your API key from https://wandb.ai/authorize
```

Or set in `~/.bashrc`:
```bash
export WANDB_API_KEY="your-key-here"
```

Training will be visible at: https://wandb.ai → project "rlt-ur5e"
