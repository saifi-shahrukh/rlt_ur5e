#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# HPC One-Time Setup
# Run on: hpc-headnode.iis.fhg.de
#
# NOTE: CentOS 7 headnode has glibc 2.17, but PyTorch 2.7.1 needs glibc 2.28+.
# Solution: Install inside a compute node (srun) OR use conda.
# This script handles both approaches.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

echo "══════════════════���════════════════════════════════════════════"
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
    exit 1
fi
echo "  ✓ Host: $(hostname) | User: $(whoami)"
echo "  ✓ BeeGFS: ${BEEGFS}"
echo "  ✓ glibc: $(ldd --version 2>&1 | head -1)"
echo ""

# ─── Step 2: Check/clone repo ────────────────────────────────────────────────
echo "[2/6] Project repository..."
if [[ -d "${PROJECT}" ]]; then
    echo "  ✓ Repo exists. Pulling latest..."
    cd "${PROJECT}" && git pull 2>/dev/null || echo "  ⚠ git pull failed"
else
    echo "  → Cloning..."
    cd "${BEEGFS}"
    git clone https://github.com/saifi-shahrukh/rlt_ur5e.git
fi
echo "  ✓ Project: ${PROJECT}"
echo ""

# ─── Step 3: Install UV ──────────────────────────────────────────────────────
echo "[3/6] UV package manager..."
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
if command -v uv &> /dev/null; then
    echo "  ✓ Already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    echo "  ✓ Installed: $(uv --version)"
fi
# Ensure in bashrc
if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"' >> ~/.bashrc
fi
echo ""

# ─── Step 4: Install Python 3.11 via uv ─────────────────────────────────────
echo "[4/6] Python & Virtual environment..."
if ! command -v python3.11 &> /dev/null; then
    echo "  → Installing Python 3.11 via uv..."
    uv python install 3.11
    echo "  ✓ Python 3.11 installed"
else
    echo "  ✓ Python 3.11 found: $(python3.11 --version)"
fi
echo ""

# Check glibc version to decide install strategy
GLIBC_VER=$(ldd --version 2>&1 | head -1 | grep -oP '\d+\.\d+$' || echo "2.17")
echo "  System glibc: ${GLIBC_VER}"

if [[ -d "${OPENPI}/.venv" && -f "${OPENPI}/.venv/bin/python" ]]; then
    echo "  ✓ Venv already exists at ${OPENPI}/.venv"
    echo "  ✓ Python: $(${OPENPI}/.venv/bin/python --version 2>/dev/null || echo 'unknown')"
else
    # glibc 2.17 (CentOS 7) can't install torch 2.7.1 directly
    # Need to install on a compute node or use conda
    echo ""
    echo "  ⚠ glibc ${GLIBC_VER} detected (CentOS 7 headnode)"
    echo "  PyTorch 2.7.1 requires glibc ≥ 2.28"
    echo ""
    echo "  Two options:"
    echo "  A) Install on a compute node (recommended if nodes have newer OS)"
    echo "  B) Install via conda/micromamba (works regardless)"
    echo ""
    echo "  Trying Option A: submitting install job to compute node..."
    echo ""

    # Create install script that runs on compute node
    cat > /tmp/hpc_install_venv.sh << 'INSTALL_EOF'
#!/bin/bash
set -e
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
cd "${OPENPI}"

echo "Installing on node: $(hostname)"
echo "glibc: $(ldd --version 2>&1 | head -1)"
echo ""

# Create venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install openpi
echo "Installing openpi (this takes 5-10 minutes)..."
uv pip install -e .

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import jax; print(f'JAX {jax.__version__} | Devices: {jax.devices()}')"

echo ""
echo "✓ Installation complete!"
INSTALL_EOF
    chmod +x /tmp/hpc_install_venv.sh

    echo "  Submitting install job (will run on a GPU node)..."
    echo "  This takes ~5-10 minutes. Check with: squeue -u saifi"
    echo ""

    # Submit as interactive or batch
    if [[ "${1}" == "--interactive" ]]; then
        echo "  Running interactively on compute node..."
        srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
            bash /tmp/hpc_install_venv.sh
    else
        # Submit as batch job
        sbatch --job-name=setup_venv \
            --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
            --output=${LOG_DIR}/setup_venv_%j.out --error=${LOG_DIR}/setup_venv_%j.err \
            /tmp/hpc_install_venv.sh
        echo "  ✓ Install job submitted! Monitor with:"
        echo "    squeue -u saifi"
        echo "    tail -f ${LOG_DIR}/setup_venv_*.out"
        echo ""
        echo "  Once it finishes, re-run this script to complete setup."
        echo "  Or run: bash 01_setup.sh --check"
    fi
fi
echo ""

# ─── Step 5: Setup directories & symlinks ────────────────────────────────────
echo "[5/6] Dataset paths & symlinks..."
mkdir -p "${DATASET_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname ${HF_SYMLINK})"

if [[ -L "${HF_SYMLINK}" ]]; then
    echo "  ✓ Symlink exists"
else
    ln -sf "${DATASET_DIR}" "${HF_SYMLINK}"
    echo "  ✓ Created: ${HF_SYMLINK} → ${DATASET_DIR}"
fi
echo ""

# ─── Step 6: W&B setup ───────────────────────────────────────────────────────
echo "[6/6] Weights & Biases..."
if [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  ✓ W&B already configured"
else
    echo "  → W&B not yet configured."
    echo ""
    echo "  After venv is installed, run:"
    echo "    source ${OPENPI}/.venv/bin/activate"
    echo "    wandb login"
    echo "  Get key at: https://wandb.ai/authorize"
fi

# Add WANDB_PROJECT to bashrc
if ! grep -q 'WANDB_PROJECT' ~/.bashrc 2>/dev/null; then
    echo 'export WANDB_PROJECT="rlt-ur5e"' >> ~/.bashrc
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
if [[ -f "${OPENPI}/.venv/bin/python" ]]; then
    echo "  ✓ SETUP COMPLETE — Ready to train!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Next steps:"
    echo "    1. Transfer dataset: bash 02_transfer_dataset.sh (from LOCAL)"
    echo "    2. W&B login:        source ${OPENPI}/.venv/bin/activate && wandb login"
    echo "    3. Train:            bash 03_train.sh both"
else
    echo "  ⏳ SETUP IN PROGRESS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Venv install is running on a compute node."
    echo "  Monitor: squeue -u saifi"
    echo "  When done: bash 01_setup.sh --check"
fi
echo ""
