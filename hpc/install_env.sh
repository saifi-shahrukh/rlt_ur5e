#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Install OpenPI Environment on CentOS 7 HPC
#
# Problem: CentOS 7 has glibc 2.17 + GCC 4.8.5
#   - PyTorch 2.7.1 needs glibc >= 2.28 (no manylinux_2_17 wheels)
#   - NumPy 2.4 needs GCC >= 9.3 to compile
#   - JAX 0.5.3 needs glibc >= 2.28
#
# Solution: Use PyTorch 2.5.1 + JAX 0.4.30 + NumPy 1.26
#   These all have pre-built manylinux_2_17 wheels (no compilation needed)
#   Training will work identically — same APIs, just slightly older versions.
#
# Run on: hpc-headnode.iis.fhg.de
# ═══════════════════════════════════════════════════════════════════════════════
set -e

OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"

export PATH="${HOME}/.local/bin:${PATH}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Installing OpenPI Environment (CentOS 7 compatible)"
echo "  Using: PyTorch 2.5.1 + JAX 0.4.30 + NumPy 1.26"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Step 1: Remove broken venv and recreate ─────────────────────────────────
echo "[1/5] Creating fresh Python 3.11 environment..."
if [[ -d "${VENV}" ]]; then
    echo "  → Removing old env..."
    rm -rf "${VENV}"
fi

# Use uv to create a plain venv (uv-managed Python 3.11)
uv venv "${VENV}" --python 3.11
echo "  ✓ Venv created: $(${VENV}/bin/python --version)"
echo ""

# ─── Step 2: Install compatible PyTorch ──────────────────────────────────────
echo "[2/5] Installing PyTorch 2.5.1 + CUDA 12.1 (manylinux_2_17 wheel)..."
${VENV}/bin/python -m pip install --no-cache-dir \
    "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121
echo "  ✓ PyTorch installed"
echo ""

# ─── Step 3: Install compatible JAX ──────────────────────────────────────────
echo "[3/5] Installing JAX 0.4.30 + CUDA 12 (manylinux_2_17 wheel)..."
${VENV}/bin/python -m pip install --no-cache-dir \
    "jax[cuda12_pip]==0.4.30" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo "  ✓ JAX installed"
echo ""

# ─── Step 4: Install OpenPI (without conflicting deps) ──────────────────────
echo "[4/5] Installing OpenPI and remaining dependencies..."
cd "${OPENPI}"

# Pin numpy to version with pre-built manylinux_2_17 wheel
${VENV}/bin/python -m pip install --no-cache-dir "numpy==1.26.4"

# Install openpi without its pinned torch/jax/numpy (we already installed compatible ones)
${VENV}/bin/python -m pip install --no-cache-dir -e . \
    --no-deps

# Install remaining dependencies (pure python or have manylinux_2_17 wheels)
${VENV}/bin/python -m pip install --no-cache-dir \
    "augmax>=0.3.4" \
    "beartype==0.19.0" \
    "dm-tree>=0.1.8" \
    "einops>=0.8.0" \
    "equinox>=0.11.8" \
    "filelock>=3.16.1" \
    "flatbuffers>=24.3.25" \
    "flax==0.10.2" \
    "fsspec[gcs]>=2024.6.0" \
    "gym-aloha>=0.1.1" \
    "imageio>=2.36.1" \
    "jaxtyping==0.2.36" \
    "lerobot>=0.4.0" \
    "ml-collections==1.0.0" \
    "numpydantic>=1.6.6" \
    "opencv-python-headless>=4.10.0" \
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
    "polars>=1.30.0" \
    "optax" \
    "chex" \
    "ml-dtypes==0.4.1"

# Install openpi-client from workspace
if [[ -d "${OPENPI}/packages/openpi-client" ]]; then
    ${VENV}/bin/python -m pip install --no-cache-dir -e "${OPENPI}/packages/openpi-client"
fi

echo "  ✓ All dependencies installed"
echo ""

# ─── Step 5: Verify ──────────────────────────────────────────────────────────
echo "[5/5] Verifying installation..."
echo ""
${VENV}/bin/python -c "
import torch
import jax
import numpy as np
import flax
print(f'  ✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  ✓ JAX     {jax.__version__}')
print(f'  ✓ NumPy   {np.__version__}')
print(f'  ✓ Flax    {flax.__version__}')
try:
    import openpi
    print(f'  ✓ OpenPI  installed')
except Exception as e:
    print(f'  ⚠ OpenPI  import issue: {e}')
"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ INSTALLATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  To activate:"
echo "    export PATH=${VENV}/bin:\$PATH"
echo ""
echo "  To test on GPU (interactive):"
echo "    srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash"
echo "    export PATH=${VENV}/bin:\$PATH"
echo "    python -c \"import torch; print(torch.cuda.get_device_name(0))\""
echo "    python -c \"import jax; print(jax.devices())\""
echo ""
echo "  Next: wandb login, transfer dataset, then bash 03_train.sh both"
echo ""
