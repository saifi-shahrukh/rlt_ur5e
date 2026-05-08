#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Install OpenPI Environment on CentOS 7 HPC
#
# CentOS 7: glibc 2.17 + GCC 4.8.5
# Solution: Use uv pip (handles everything) + older compatible wheels
#   - PyTorch 2.5.1 (manylinux_2_17 wheel)
#   - JAX 0.4.30 (manylinux_2_17 wheel)
#   - NumPy 1.26.4 (pre-built wheel)
#
# Run on: hpc-headnode.iis.fhg.de
# ═══════════════════════════════════════════════════════════════════════════════
set -e

OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Installing OpenPI Environment (CentOS 7 compatible)"
echo "  Using: uv pip + PyTorch 2.5.1 + JAX 0.4.30 + NumPy 1.26"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Step 1: Remove broken venv and recreate ─────────────────────────────────
echo "[1/5] Creating fresh Python 3.11 environment..."
if [[ -d "${VENV}" ]]; then
    echo "  → Removing old env..."
    rm -rf "${VENV}"
fi

uv venv "${VENV}" --python 3.11 --seed
echo "  ✓ Venv created: $(${VENV}/bin/python --version)"
echo ""

# ─── Step 2: Install compatible PyTorch ──────────────────────────────────────
echo "[2/5] Installing PyTorch 2.5.1 + CUDA 12.1 (manylinux_2_17 wheel)..."
uv pip install --python "${VENV}/bin/python" \
    "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121
echo "  ✓ PyTorch installed"
echo ""

# ─── Step 3: Install compatible JAX ──────────────────────────────────────────
echo "[3/5] Installing JAX 0.4.30 + CUDA 12..."
uv pip install --python "${VENV}/bin/python" \
    "jax[cuda12_pip]==0.4.30" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo "  ✓ JAX installed"
echo ""

# ─── Step 4: Install OpenPI + all dependencies ───────────────────────────────
echo "[4/5] Installing OpenPI and remaining dependencies..."
cd "${OPENPI}"

# Pin numpy first (pre-built wheel, no compilation)
uv pip install --python "${VENV}/bin/python" "numpy==1.26.4"

# Install openpi in no-deps mode (we handle deps manually)
uv pip install --python "${VENV}/bin/python" -e . --no-deps

# Install remaining deps (all have manylinux_2_17 wheels or are pure python)
uv pip install --python "${VENV}/bin/python" \
    "augmax>=0.3.4" \
    "beartype==0.19.0" \
    "dm-tree>=0.1.8" \
    "einops>=0.8.0" \
    "equinox>=0.11.8" \
    "filelock>=3.16.1" \
    "flatbuffers>=24.3.25" \
    "flax==0.10.2" \
    "fsspec>=2024.6.0" \
    "gcsfs>=2024.6.0" \
    "gym-aloha>=0.1.1" \
    "imageio>=2.36.1" \
    "jaxtyping==0.2.36" \
    "lerobot>=0.4.0" \
    "ml-collections==1.0.0" \
    "numpydantic>=1.6.6" \
    "opencv-python-headless>=4.8.0" \
    "orbax-checkpoint>=0.6" \
    "pillow>=11.0.0" \
    "sentencepiece>=0.2.0" \
    "tqdm-loggable>=0.2" \
    "typing-extensions>=4.12.2" \
    "tyro>=0.9.5" \
    "wandb>=0.19.1" \
    "treescope>=0.1.7" \
    "transformers>=4.40" \
    "rich>=14.0.0" \
    "polars>=1.0.0" \
    "optax" \
    "chex" \
    "ml-dtypes==0.4.1"

# Install openpi-client from workspace
if [[ -d "${OPENPI}/packages/openpi-client" ]]; then
    uv pip install --python "${VENV}/bin/python" -e "${OPENPI}/packages/openpi-client"
fi

echo "  ✓ All dependencies installed"
echo ""

# ─── Step 5: Verify ──────────────────────────────────────────────────────────
echo "[5/5] Verifying installation..."
echo ""
${VENV}/bin/python << 'PYEOF'
import sys
try:
    import torch
    print(f'  ✓ PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
except Exception as e:
    print(f'  ✗ PyTorch FAILED: {e}')
    sys.exit(1)

try:
    import jax
    print(f'  ✓ JAX     {jax.__version__}')
except Exception as e:
    print(f'  ✗ JAX FAILED: {e}')

try:
    import numpy as np
    print(f'  ✓ NumPy   {np.__version__}')
except Exception as e:
    print(f'  ✗ NumPy FAILED: {e}')

try:
    import flax
    print(f'  ✓ Flax    {flax.__version__}')
except Exception as e:
    print(f'  ✗ Flax FAILED: {e}')

try:
    import wandb
    print(f'  ✓ W&B     {wandb.__version__}')
except Exception as e:
    print(f'  ✗ W&B FAILED: {e}')

try:
    import openpi
    print(f'  ✓ OpenPI  OK')
except Exception as e:
    print(f'  ⚠ OpenPI  import issue: {e}')
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ INSTALLATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Activate:  export PATH=${VENV}/bin:\$PATH"
echo ""
echo "  W&B login: ${VENV}/bin/wandb login"
echo ""
echo "  Test GPU:  srun --partition=gpu --gres=gpu:1 --time=00:05:00 --pty bash"
echo "             export PATH=${VENV}/bin:\$PATH"
echo "             python -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo ""
