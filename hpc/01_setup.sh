#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# HPC One-Time Setup
# Run on: hpc-headnode.iis.fhg.de
# ═══════════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Setup — π0/π0.5 Fine-tuning"
echo "  Cluster: Fraunhofer IIS HPC (V100 32GB)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Configuration ───────────────────────────────────────────────────────────
BEEGFS="/data/beegfs/home/saifi"
PROJECT="${BEEGFS}/rlt_ur5e"
OPENPI="${PROJECT}/openpi_ur5e/openpi-ur5e"
DATASET_DIR="${BEEGFS}/datasets/saifi/ur5e-peg-insertion-dual"
HF_SYMLINK="${HOME}/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual"
LOG_DIR="${BEEGFS}/logs"

# ─── Step 1: Verify location ─────────────────────────────────────────────────
echo "[1/6] Checking environment..."
if [[ ! -d "${BEEGFS}" ]]; then
    echo "  ERROR: BeeGFS not found at ${BEEGFS}"
    echo "  Are you on hpc-headnode?"
    exit 1
fi
echo "  ✓ Host: $(hostname) | User: $(whoami)"
echo "  ✓ BeeGFS: ${BEEGFS}"
echo ""

# ─── Step 2: Check/clone repo ────────────────────────────────────────────────
echo "[2/6] Project repository..."
if [[ -d "${PROJECT}" ]]; then
    echo "  ✓ Repo exists. Pulling latest..."
    cd "${PROJECT}" && git pull 2>/dev/null || echo "  ⚠ git pull failed (local changes?)"
else
    echo "  → Cloning..."
    cd "${BEEGFS}"
    git clone git@github.com:saifi-shahrukh/rlt_ur5e.git
fi
echo "  ✓ Project: ${PROJECT}"
echo ""

# ─── Step 3: Install UV ──────────────────────────────────────────────────────
echo "[3/6] UV package manager..."
if command -v uv &> /dev/null; then
    echo "  ✓ Already installed: $(uv --version)"
else
    echo "  → Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    # Add to bashrc for future sessions
    if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"' >> ~/.bashrc
    fi
    echo "  ✓ Installed: $(uv --version)"
fi
echo ""

# ─── Step 4: Create venv ─────────────────────────────────────────────────────
echo "[4/6] Virtual environment..."
cd "${OPENPI}"
if [[ -d ".venv" && -f ".venv/bin/python" ]]; then
    echo "  ✓ Venv exists at ${OPENPI}/.venv"
    echo "  ✓ Python: $(.venv/bin/python --version)"
else
    echo "  → Finding Python 3.11+..."
    PYTHON_BIN=""
    for p in python3.11 python3.12 python3.10; do
        if command -v $p &> /dev/null; then
            PYTHON_BIN=$p
            break
        fi
    done
    # Try module system
    if [[ -z "${PYTHON_BIN}" ]] && command -v module &> /dev/null; then
        module load python/3.11 2>/dev/null || module load Python/3.11 2>/dev/null || true
        for p in python3.11 python3.12 python3.10; do
            if command -v $p &> /dev/null; then
                PYTHON_BIN=$p
                break
            fi
        done
    fi
    if [[ -z "${PYTHON_BIN}" ]]; then
        echo "  ERROR: No Python 3.10+ found!"
        echo "  Try: module avail | grep -i python"
        exit 1
    fi
    echo "  → Creating venv with ${PYTHON_BIN}..."
    uv venv .venv --python ${PYTHON_BIN}
    echo "  → Installing openpi (this may take a few minutes)..."
    source .venv/bin/activate
    uv pip install -e .
    echo "  ✓ Venv created and openpi installed"
fi
echo ""

# ─── Step 5: Setup directories & symlinks ────────────────────────────────────
echo "[5/6] Dataset paths & symlinks..."
mkdir -p "${DATASET_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname ${HF_SYMLINK})"

if [[ -L "${HF_SYMLINK}" ]]; then
    echo "  ✓ Symlink exists: ${HF_SYMLINK}"
else
    ln -sf "${DATASET_DIR}" "${HF_SYMLINK}"
    echo "  ✓ Created: ${HF_SYMLINK} → ${DATASET_DIR}"
fi
echo "  ✓ Logs dir: ${LOG_DIR}"
echo ""

# ─── Step 6: W&B setup (for live training visibility) ────────────────────────
echo "[6/6] Weights & Biases (online logging)..."
source "${OPENPI}/.venv/bin/activate"
if [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  ✓ W&B already configured — training will be visible online"
else
    echo "  → W&B not configured. Setting up now..."
    echo ""
    echo "  You need your API key from: https://wandb.ai/authorize"
    echo ""
    # Try to login interactively
    if [[ -t 0 ]]; then
        wandb login
    else
        echo "  Non-interactive shell. Run manually:"
        echo "    source ${OPENPI}/.venv/bin/activate"
        echo "    wandb login"
        echo ""
        echo "  Or add to ~/.bashrc:"
        echo "    export WANDB_API_KEY=<your-key>"
    fi
fi
# Set default W&B project
if ! grep -q 'WANDB_PROJECT' ~/.bashrc 2>/dev/null; then
    echo 'export WANDB_PROJECT="rlt-ur5e"' >> ~/.bashrc
    echo "  ✓ Added WANDB_PROJECT=rlt-ur5e to ~/.bashrc"
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Project:   ${PROJECT}"
echo "  OpenPI:    ${OPENPI}"
echo "  Venv:      ${OPENPI}/.venv"
echo "  Dataset:   ${DATASET_DIR}"
echo "  Logs:      ${LOG_DIR}"
echo ""

# Check if dataset has files
if [[ -z "$(ls -A ${DATASET_DIR} 2>/dev/null)" ]]; then
    echo "  ⚠ Dataset is EMPTY — transfer from local machine:"
    echo "    bash 02_transfer_dataset.sh"
else
    NUM=$(ls ${DATASET_DIR} | wc -l)
    echo "  ✓ Dataset: ${NUM} items found"
fi
echo ""
echo "  Next: Transfer dataset → bash 02_transfer_dataset.sh (from LOCAL)"
echo "         Then train     → bash 03_train.sh all"
echo ""
