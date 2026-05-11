#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Fix pip resolution-too-deep error
#
# Strategy: Install dependencies in stages with pinned versions to avoid
# pip's resolver getting overwhelmed by the dependency graph.
#
# Run on HPC after torch is already installed:
#   cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e
#   export PATH=".venv/bin:$PATH"
#   bash /data/beegfs/home/saifi/rlt_ur5e/hpc/fix_resolution.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

VENV="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv"
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
export PATH="${VENV}/bin:${PATH}"

cd "${OPENPI}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Staged Installation (avoiding resolution-too-deep)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo ""

# ─── Stage 1: Core scientific packages ────────────────────────────────────────
echo "[1/7] Core scientific packages..."
pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy>=1.11.1" \
    "pillow>=11.0.0" \
    "PyYAML>=6.0" \
    "ml-dtypes==0.4.1" \
    2>&1 | tail -5
echo ""

# ─── Stage 2: JAX ecosystem ───────────────────────────────────────────────────
echo "[2/7] JAX + Flax + Orbax..."
pip install --no-cache-dir \
    "jax[cuda12]==0.5.3" \
    "flax==0.10.2" \
    "optax" \
    "orbax-checkpoint==0.11.13" \
    2>&1 | tail -5
echo ""

# ─── Stage 3: HuggingFace ecosystem (pinned to avoid backtracking) ────────────
echo "[3/7] HuggingFace: transformers, datasets, diffusers..."
pip install --no-cache-dir \
    "huggingface-hub>=0.35.0,<0.36.0" \
    "transformers==4.53.2" \
    "safetensors>=0.6.0" \
    "tokenizers>=0.21,<0.22" \
    "datasets>=4.2.0,<4.9.0" \
    "diffusers>=0.30.0,<0.36.0" \
    "accelerate>=1.10.0,<2.0.0" \
    2>&1 | tail -5
echo ""

# ─── Stage 4: torchvision + torchcodec (pinned for torch 2.7.1) ───────────────
echo "[4/7] torchvision + torchcodec..."
pip install --no-cache-dir \
    "torchvision==0.22.1" \
    --index-url https://download.pytorch.org/whl/cu126 \
    2>&1 | tail -5

pip install --no-cache-dir \
    "torchcodec>=0.2.1" \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    2>&1 | tail -5
echo ""

# ─── Stage 5: lerobot + its tricky deps (no gym-aloha) ───────────────────────
echo "[5/7] lerobot (without rerun-sdk)..."
# Install lerobot's mandatory deps that cause backtracking
pip install --no-cache-dir \
    "pyarrow>=21.0" \
    "av>=15.0.0,<16.0.0" \
    "opencv-python-headless>=4.9.0,<4.13.0" \
    "draccus==0.10.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "cmake>=3.29.0,<4.2.0" \
    "pynput>=1.7.7,<1.9.0" \
    2>&1 | tail -5

# Install lerobot itself (--no-deps to skip rerun-sdk)
pip install --no-cache-dir "lerobot>=0.4.0" --no-deps 2>&1 | tail -3
echo ""

# ─── Stage 6: gym-aloha + mujoco (pinned) ────────────────────────────────────
echo "[6/7] gym-aloha + gymnasium + mujoco..."
pip install --no-cache-dir \
    "gymnasium>=1.0.0,<1.4.0" \
    "mujoco>=3.0.0" \
    "dm-control>=1.0.15" \
    "gym-aloha>=0.1.1" \
    2>&1 | tail -5
echo ""

# ─── Stage 7: Remaining OpenPI deps + editable install ───────────────────────
echo "[7/7] OpenPI remaining deps + editable install..."
pip install --no-cache-dir \
    "augmax>=0.3.4" \
    "beartype==0.19.0" \
    "dm-tree>=0.1.8" \
    "einops>=0.8.0" \
    "equinox>=0.11.8" \
    "flatbuffers>=24.3.25" \
    "fsspec[gcs]>=2024.6.0" \
    "imageio>=2.36.1" \
    "jaxtyping==0.2.36" \
    "ml-collections==1.0.0" \
    "numpydantic>=1.6.6" \
    "opencv-python>=4.10.0.84" \
    "polars>=1.30.0" \
    "rich>=14.0.0" \
    "sentencepiece>=0.2.0" \
    "tqdm-loggable>=0.2" \
    "treescope>=0.1.7" \
    "tyro>=0.9.5" \
    "wandb>=0.19.1" \
    2>&1 | tail -5
echo ""

# Final: Install openpi itself as editable (no-deps since everything is in place)
echo "  Installing openpi (editable, --no-deps)..."
pip install --no-cache-dir -e . --no-deps 2>&1 | tail -3

# Install openpi-client
if [[ -d "${OPENPI}/packages/openpi-client" ]]; then
    pip install --no-cache-dir -e "${OPENPI}/packages/openpi-client" --no-deps 2>&1 | tail -3
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ INSTALLATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Verify ──────────────────────────────────────────────────────────────────
echo "Verifying key imports..."
python << 'EOF'
import sys
failed = []
for pkg in ["torch", "jax", "jaxlib", "flax", "optax", "numpy", 
            "orbax.checkpoint", "wandb", "transformers", "polars",
            "sentencepiece", "lerobot", "openpi", "openpi.training.config"]:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except Exception as e:
        print(f"  ✗ {pkg}: {e}")
        failed.append(pkg)

if failed:
    print(f"\n  ⚠ Failed: {failed}")
    sys.exit(1)
else:
    print("\n  🎉 ALL IMPORTS PASS!")
EOF

echo ""
echo "Next: Set up activate_hpc.sh and test training"
echo "  source ${VENV}/activate_hpc.sh"
echo "  python scripts/train.py pi0_ur5e_peg_insertion_lora --help"
