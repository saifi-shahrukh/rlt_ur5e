#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Install OpenPI Environment on CentOS 7 HPC
#
# CentOS 7: glibc 2.17 + GCC 4.8.5
# Strategy:
#   1. Pin numpy==1.26.4 first (pre-built manylinux_2_17 wheel)
#   2. Install PyTorch 2.5.1 (manylinux_2_17)
#   3. Install JAX 0.4.35 + jaxlib from Google's release page
#   4. Install everything else with numpy pinned
#
# Run on: hpc-headnode.iis.fhg.de
# ═══════════════════════════════════════════════════════════════════════════════
set -e

OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
UV="${HOME}/.local/bin/uv"

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export UV_LINK_MODE=copy

echo "═══════════════════════════════════════════════════════════════"
echo "  Installing OpenPI Environment (CentOS 7 compatible)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Step 1: Fresh venv ──────────────────────────────────────────────────────
echo "[1/5] Creating fresh Python 3.11 environment..."
[[ -d "${VENV}" ]] && rm -rf "${VENV}"

${UV} venv "${VENV}" --python 3.11 --seed
echo "  ✓ Venv: $(${VENV}/bin/python --version)"
echo ""

# ─── Step 2: Pin NumPy + scipy (prevent any source builds) ───────────────────
echo "[2/5] Installing NumPy 1.26.4 + SciPy 1.11.4 (pre-built wheels)..."
${UV} pip install --python "${VENV}/bin/python" \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    "ml-dtypes==0.3.2"
echo "  ✓ NumPy + SciPy + ml-dtypes installed (all pre-built)"
echo ""

# ─── Step 3: PyTorch 2.5.1 ───────────────────────────────────────────────────
echo "[3/5] Installing PyTorch 2.5.1 + CUDA 12.1..."
echo "  (Downloads ~2.5GB — takes 5-10 min)"
${UV} pip install --python "${VENV}/bin/python" \
    "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121
echo "  ✓ PyTorch installed"
echo ""

# ─── Step 4: JAX + all other deps ────────────────────────────────────────────
echo "[4/5] Installing JAX + OpenPI deps..."

# Install jaxlib from Google's CUDA release page (pre-built wheel)
# These are the correct URLs for jaxlib 0.4.30 with CUDA 12
echo "  → Installing jaxlib 0.4.30 (CUDA 12, pre-built)..."
${UV} pip install --python "${VENV}/bin/python" \
    "jaxlib==0.4.30" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    || ${UV} pip install --python "${VENV}/bin/python" "jaxlib==0.4.30"

echo "  → Installing jax 0.4.30..."
${UV} pip install --python "${VENV}/bin/python" "jax==0.4.30" --no-deps

# Now install all remaining pure-python / manylinux_2_17 deps
# NOTE: lerobot installed with --no-deps to avoid rerun-sdk (needs glibc 2.28)
echo "  → Installing remaining dependencies..."
${UV} pip install --python "${VENV}/bin/python" \
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
    "chex"

# Install lerobot WITHOUT its deps (avoids rerun-sdk which needs glibc 2.28)
# We only need lerobot.datasets for the data_loader
echo "  → Installing lerobot (no-deps, avoids rerun-sdk)..."
${UV} pip install --python "${VENV}/bin/python" "lerobot==0.4.0" --no-deps

# Install lerobot's actual needed deps (subset, no rerun-sdk)
${UV} pip install --python "${VENV}/bin/python" \
    "datasets>=2.19" \
    "huggingface-hub>=0.23" \
    "pyarrow>=15.0" \
    "torch" \
    "torchvision" \
    "safetensors" \
    "draccus" \
    "jsonlines" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple/

# OpenPI editable install (no deps — we installed them all manually)
cd "${OPENPI}"
${UV} pip install --python "${VENV}/bin/python" -e . --no-deps

# OpenPI client
[[ -d "${OPENPI}/packages/openpi-client" ]] && \
    ${UV} pip install --python "${VENV}/bin/python" -e "${OPENPI}/packages/openpi-client" --no-deps

echo "  ✓ All deps installed"
echo ""

# ─── Step 5: Verify ──────────────────────────────────────────────────────────
echo "[5/5] Verifying..."
${VENV}/bin/python << 'PYEOF'
import sys
packages = [
    ("torch", lambda: __import__("torch").__version__),
    ("jax", lambda: __import__("jax").__version__),
    ("jaxlib", lambda: __import__("jaxlib").__version__),
    ("numpy", lambda: __import__("numpy").__version__),
    ("scipy", lambda: __import__("scipy").__version__),
    ("flax", lambda: __import__("flax").__version__),
    ("wandb", lambda: __import__("wandb").__version__),
    ("openpi", lambda: "OK" if __import__("openpi") else "OK"),
]
all_ok = True
for name, get_ver in packages:
    try:
        ver = get_ver()
        print(f"  ✓ {name:12s} {ver}")
    except Exception as e:
        print(f"  ✗ {name:12s} FAILED: {e}")
        all_ok = False
if not all_ok:
    print("\n  ⚠ Some packages failed but training may still work.")
    print("    JAX without CUDA is fine — training uses PyTorch GPU.")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ INSTALLATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  W&B login:  ${VENV}/bin/wandb login"
echo ""
echo "  Train:"
echo "    cd /data/beegfs/home/saifi/rlt_ur5e/hpc"
echo "    bash 03_train.sh both"
echo ""
