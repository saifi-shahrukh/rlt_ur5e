#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# HPC One-Time Setup
# Run on: hpc-headnode.iis.fhg.de
#
# Problem: CentOS 7 has glibc 2.17, PyTorch 2.7.1 needs glibc 2.28+
# Solution: Use conda (micromamba) which bundles its own glibc/libs
# ═══════════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Setup — π0/π0.5 Fine-tuning"
echo "  Cluster: Fraunhofer IIS HPC (V100 32GB, CentOS 7)"
echo "  Strategy: micromamba + conda-forge (bypasses glibc 2.17)"
echo "═══════════��═══════════════════════════════════════════════════"
echo ""

# ─── Configuration ───────────────────────────────────────────────────────────
BEEGFS="/data/beegfs/home/saifi"
PROJECT="${BEEGFS}/rlt_ur5e"
OPENPI="${PROJECT}/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
DATASET_DIR="${BEEGFS}/datasets/saifi/ur5e-peg-insertion-dual"
HF_SYMLINK="${HOME}/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual"
LOG_DIR="${BEEGFS}/logs"
MICROMAMBA="${HOME}/.local/bin/micromamba"

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

# ─── Step 1: Verify location ─────────────────────────────────────────────────
echo "[1/7] Checking environment..."
if [[ ! -d "${BEEGFS}" ]]; then
    echo "  ERROR: BeeGFS not found at ${BEEGFS}"
    exit 1
fi
echo "  ✓ Host: $(hostname) | User: $(whoami)"
echo "  ✓ glibc: $(ldd --version 2>&1 | head -1)"
echo ""

# ─── Step 2: Check/clone repo ────────────────────────────────────────────────
echo "[2/7] Project repository..."
if [[ -d "${PROJECT}" ]]; then
    cd "${PROJECT}" && git pull 2>/dev/null || echo "  ⚠ git pull failed"
    echo "  ✓ Project: ${PROJECT}"
else
    cd "${BEEGFS}"
    git clone https://github.com/saifi-shahrukh/rlt_ur5e.git
    echo "  ✓ Cloned: ${PROJECT}"
fi
echo ""

# ─── Step 3: Install micromamba ──────────────────────────────────────────────
echo "[3/7] Micromamba (conda replacement)..."
if [[ -f "${MICROMAMBA}" ]]; then
    echo "  ✓ Already installed: $(${MICROMAMBA} --version)"
else
    echo "  → Installing micromamba..."
    mkdir -p "${HOME}/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${HOME}/.local/bin" --strip-components=1 bin/micromamba
    chmod +x "${MICROMAMBA}"
    echo "  ✓ Installed: $(${MICROMAMBA} --version)"
fi
echo ""

# ─── Step 4: Create conda environment ───────────────────────────────────────
echo "[4/7] Conda environment (Python 3.11 + CUDA)..."
if [[ -f "${VENV}/bin/python" ]]; then
    echo "  ✓ Environment exists at ${VENV}"
    echo "  ✓ Python: $(${VENV}/bin/python --version)"
else
    echo "  → Creating conda env at ${VENV}..."
    echo "  (This takes 3-5 minutes — downloading Python + CUDA toolkit)"
    echo ""
    ${MICROMAMBA} create -p "${VENV}" \
        python=3.11 \
        pip \
        cuda-toolkit=12.4 \
        -c conda-forge -c nvidia \
        -y
    echo ""
    echo "  ✓ Conda env created with Python 3.11 + CUDA 12.4"
fi
echo ""

# ─── Step 5: Install OpenPI ──────────────────────────────────────────────────
echo "[5/7] Installing OpenPI..."
# Activate — use full path to ensure we find pip
export PATH="${VENV}/bin:${PATH}"
export CONDA_PREFIX="${VENV}"

# Verify pip is accessible
if [[ ! -f "${VENV}/bin/pip" ]]; then
    echo "  ⚠ pip not found in conda env. Installing..."
    ${VENV}/bin/python -m ensurepip --upgrade 2>/dev/null || true
fi

# Check if openpi already installed
if ${VENV}/bin/python -c "import openpi" 2>/dev/null; then
    echo "  ✓ OpenPI already installed"
else
    echo "  → Installing openpi and dependencies..."
    echo "  (This takes 5-10 minutes — PyTorch, JAX, etc.)"
    echo ""
    cd "${OPENPI}"
    
    # Install with pip using full path (conda provides the glibc compatibility layer)
    ${VENV}/bin/pip install --no-cache-dir -e . 2>&1 | tail -10
    
    echo ""
    echo "  → Verifying installation..."
    ${VENV}/bin/python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || echo "  ⚠ PyTorch import failed"
    ${VENV}/bin/python -c "import jax; print(f'  ✓ JAX {jax.__version__}')" || echo "  ⚠ JAX import failed"
    echo "  ✓ OpenPI installed"
fi
echo ""

# ─── Step 6: Setup directories & symlinks ────────────────────────────────────
echo "[6/7] Dataset paths & symlinks..."
mkdir -p "${DATASET_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname ${HF_SYMLINK})"

if [[ -L "${HF_SYMLINK}" ]]; then
    echo "  ✓ Symlink exists"
else
    ln -sf "${DATASET_DIR}" "${HF_SYMLINK}"
    echo "  ✓ Created: ${HF_SYMLINK}"
fi
echo ""

# ─── Step 7: W&B setup ───────────────────────────────────────────────────────
echo "[7/7] Weights & Biases (live training curves)..."
if [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  ✓ W&B configured — training visible at wandb.ai"
else
    echo "  → Setting up W&B..."
    echo "    Get your API key at: https://wandb.ai/authorize"
    echo ""
    if [[ -t 0 ]]; then
        # Interactive — try login now
        export PATH="${VENV}/bin:${PATH}"
        python -m wandb login 2>/dev/null || {
            echo "  Manual login needed. Run later:"
            echo "    export PATH=${VENV}/bin:\$PATH"
            echo "    wandb login"
        }
    else
        echo "  Run after setup: wandb login"
    fi
fi

# Ensure WANDB_PROJECT in bashrc
if ! grep -q 'WANDB_PROJECT' ~/.bashrc 2>/dev/null; then
    echo 'export WANDB_PROJECT="rlt-ur5e"' >> ~/.bashrc
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════���═══════"
echo "  ✓ SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Env:       ${VENV}"
echo "  Python:    $(${VENV}/bin/python --version 2>/dev/null || echo 'pending')"
echo "  Project:   ${PROJECT}"
echo "  Dataset:   ${DATASET_DIR}"
echo "  Logs:      ${LOG_DIR}"
echo ""

if [[ -z "$(ls -A ${DATASET_DIR} 2>/dev/null)" ]]; then
    echo "  ⚠ Dataset EMPTY — transfer from local:"
    echo "    bash 02_transfer_dataset.sh (from LOCAL machine)"
    echo ""
fi

echo "  Next steps:"
echo "    1. Transfer dataset (LOCAL):  bash 02_transfer_dataset.sh"
echo "    2. Train (HPC):               cd hpc && bash 03_train.sh both"
echo "    3. Monitor:                    bash 04_status.sh"
echo "    4. Watch live:                 https://wandb.ai → project 'rlt-ur5e'"
echo ""
