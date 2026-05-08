# HPC Environment Installation — Verified Working Setup

> CentOS 7 (glibc 2.17, GCC 4.8.5) — No sudo, no containers
> Cluster: Fraunhofer IIS HPC, V100 32GB GPUs

---

## ✅ Final Working Environment

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11.15 | Installed via `uv python install 3.11` |
| PyTorch | 2.5.1+cu121 | manylinux_2_17 wheel |
| JAX | 0.4.30 | --no-deps install |
| jaxlib | 0.4.30 | from Google releases |
| Flax | 0.10.2 | --no-deps install |
| NumPy | 1.26.4 | Pre-built wheel (last manylinux_2_17) |
| SciPy | 1.11.4 | Pre-built wheel |
| W&B | 0.19.11 | Pre-built wheel |
| Transformers | 5.8.0 | |
| OpenPI | 0.1.0 | editable install |
| lerobot | 0.4.0 | --no-deps (avoids rerun-sdk) |
| datasets | 2.19.0 | --no-deps (avoids pyarrow 24) |
| pyarrow | 14.0.2 | Pre-built manylinux_2_17 |
| opencv-python-headless | 4.9.0.80 | Last manylinux_2_17 version |
| tensorstore | 0.1.45 | For orbax-checkpoint |
| sentencepiece | 0.1.99 | Pre-built (0.2.x needs cmake) |
| dm-tree | 0.1.8 | Pre-built wheel |

---

## Constraints (Why these versions)

- **glibc 2.17**: PyTorch >=2.6, JAX >=0.5, NumPy >=2.0, rerun-sdk all need glibc 2.28+
- **GCC 4.8.5**: NumPy >=2.0 needs GCC >=9.3 to compile from source
- **No cmake**: dm-tree >=0.1.9, sentencepiece >=0.2.0 need cmake
- **No Rust**: pyarrow >=15, libcst (dep of pyarrow build) need Rust compiler
- **No Go**: wandb >=0.26 needs Go to build from source

---

## Complete Install Steps (from scratch)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

# 2. Install Python 3.11
uv python install 3.11

# 3. Clone repo
cd /data/beegfs/home/saifi
git clone https://github.com/saifi-shahrukh/rlt_ur5e.git

# 4. Create venv
cd rlt_ur5e/openpi_ur5e/openpi-ur5e
uv venv .venv --python 3.11 --seed
export UV_LINK_MODE=copy
VENV="$(pwd)/.venv"
UV="${HOME}/.local/bin/uv"

# 5. Core packages (order matters!)
${UV} pip install --python "${VENV}/bin/python" "numpy==1.26.4" "scipy==1.11.4" "ml-dtypes==0.3.2"
${UV} pip install --python "${VENV}/bin/python" "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
${UV} pip install --python "${VENV}/bin/python" "jaxlib==0.4.30" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
${UV} pip install --python "${VENV}/bin/python" "jax==0.4.30" --no-deps

# 6. Packages that must be pinned (avoid source builds)
${UV} pip install --python "${VENV}/bin/python" "dm-tree==0.1.8"
${UV} pip install --python "${VENV}/bin/python" "opencv-python-headless==4.9.0.80"
${UV} pip install --python "${VENV}/bin/python" "pyarrow==14.0.2"
${UV} pip install --python "${VENV}/bin/python" "tensorstore==0.1.45"
${UV} pip install --python "${VENV}/bin/python" "sentencepiece==0.1.99"
${UV} pip install --python "${VENV}/bin/python" "wandb==0.19.11"

# 7. --no-deps packages (avoid pulling incompatible transitive deps)
${UV} pip install --python "${VENV}/bin/python" "flax==0.10.2" --no-deps
${UV} pip install --python "${VENV}/bin/python" "orbax-checkpoint==0.6.4" --no-deps
${UV} pip install --python "${VENV}/bin/python" "optax" --no-deps
${UV} pip install --python "${VENV}/bin/python" "chex" --no-deps
${UV} pip install --python "${VENV}/bin/python" "equinox" --no-deps
${UV} pip install --python "${VENV}/bin/python" "lerobot==0.4.0" --no-deps
${UV} pip install --python "${VENV}/bin/python" "datasets==2.19.0" --no-deps

# 8. Pure-python packages (no version issues)
${UV} pip install --python "${VENV}/bin/python" \
    "opt-einsum" "msgpack" "einops" "augmax" "flatbuffers" "imageio" \
    "beartype==0.19.0" "jaxtyping==0.2.36" "ml-collections==1.0.0" \
    "tyro" "tqdm-loggable" "treescope" "rich" "numpydantic" \
    "transformers>=4.40" "gcsfs" \
    "huggingface-hub>=0.23" "safetensors" "draccus" "jsonlines" \
    "dill" "multiprocess" "xxhash" "fsspec" "aiohttp" "requests" \
    "tqdm" "pyyaml" "click" "gitpython" "psutil" "sentry-sdk" \
    "docker-pycreds" "platformdirs" "protobuf" "setproctitle"

# 9. OpenPI editable install
${UV} pip install --python "${VENV}/bin/python" -e . --no-deps
${UV} pip install --python "${VENV}/bin/python" -e packages/openpi-client --no-deps

# 10. Dataset symlink
mkdir -p ~/.cache/huggingface/lerobot/saifi
ln -sf /data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual
```

---

## W&B Setup

wandb 0.19.11 doesn't support the new `wandb_v1_...` API key format (86 chars).
Use environment variable instead:

```bash
# Add to ~/.bashrc:
export WANDB_API_KEY="your-key-here"
export WANDB_PROJECT="rlt-ur5e"
```

Or set in SLURM scripts directly.

---

## Dataset Location

```
/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual/
├── data/chunk-000/file-000.parquet      (274 KB - 9 episodes)
├── meta/info.json
├── meta/stats.json  
├── meta/tasks.parquet
├── meta/episodes/chunk-000/file-000.parquet
├── videos/observation.images.overview_cam/chunk-000/file-000.mp4  (38 MB)
└── videos/observation.images.wrist_cam/chunk-000/file-000.mp4     (12 MB)
```

Symlinked to: `~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual`

---

## Submit Training

```bash
cd /data/beegfs/home/saifi/rlt_ur5e/hpc
bash 03_train.sh both    # π0 + π0.5 in parallel
```
