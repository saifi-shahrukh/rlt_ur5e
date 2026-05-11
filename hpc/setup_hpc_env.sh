#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DEFINITIVE HPC SETUP — OpenPI Fine-tuning on CentOS 7
#
# Problem:  CentOS 7 has glibc 2.17. Modern PyTorch/JAX need glibc 2.28+.
#           Using --no-deps causes cascading missing-dependency catastrophes.
#
# Solution: Install conda's sysroot_linux-64=2.28 (provides glibc 2.28 runtime),
#           then patchelf Python to USE that glibc. After patching, pip sees
#           glibc 2.28 and installs ALL packages normally WITH their full deps.
#
# Why this works:
#   1. conda's sysroot_linux-64=2.28 provides a REAL glibc 2.28 (libc.so.6,
#      ld-linux-x86-64.so.2) inside the conda env
#   2. patchelf changes Python's ELF interpreter to use sysroot's ld-linux
#   3. patchelf sets RPATH so Python loads glibc 2.28 at startup
#   4. pip detects glibc 2.28 → downloads manylinux_2_28 wheels normally
#   5. At runtime, activate_hpc.sh sets LD_LIBRARY_PATH for .so files
#   6. glibc 2.28 is backward-compatible with code built for glibc 2.17
#
# CRITICAL: Do NOT set LD_LIBRARY_PATH during installation!
#   System binaries (uname, gcc, etc.) use the system's ld-linux (glibc 2.17)
#   and will CRASH if they find glibc 2.28's libc.so.6 in LD_LIBRARY_PATH.
#   LD_LIBRARY_PATH is ONLY set at runtime via activate_hpc.sh.
#
# Run ONCE on HPC headnode:
#   cd /data/beegfs/home/saifi/rlt_ur5e/hpc && bash setup_hpc_env.sh
#
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────────
BEEGFS="/data/beegfs/home/saifi"
PROJECT="${BEEGFS}/rlt_ur5e"
OPENPI="${PROJECT}/openpi_ur5e/openpi-ur5e"
VENV="${OPENPI}/.venv"
MICROMAMBA="${HOME}/.local/bin/micromamba"
LOG_DIR="${BEEGFS}/logs"
DATASET_DIR="${BEEGFS}/datasets/saifi/ur5e-peg-insertion-dual"
HF_SYMLINK="${HOME}/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual"

export PATH="${HOME}/.local/bin:${PATH}"

# CRITICAL: Ensure LD_LIBRARY_PATH does NOT contain sysroot paths during install
unset LD_LIBRARY_PATH 2>/dev/null || true

echo "═════════════════════════════════════════════════════════════════"
echo "  HPC Setup — OpenPI π0/π0.5 Fine-tuning"
echo "  Strategy: sysroot glibc 2.28 + patchelf (no --no-deps needed!)"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "  Host:   $(hostname)"
echo "  glibc:  $(ldd --version 2>&1 | head -1 | grep -oP '\d+\.\d+$' || echo 'unknown')"
echo "  User:   $(whoami)"
echo ""

# ═════════════════════════════════════════════════════════════════
# STEP 1: Install micromamba
# ═════════════════════════════════════════════════════════════════
echo "[1/6] Micromamba..."
if [[ -f "${MICROMAMBA}" ]]; then
    echo "  ✓ Already installed: $(${MICROMAMBA} --version)"
else
    echo "  → Installing micromamba..."
    mkdir -p "${HOME}/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "${HOME}/.local/bin" --strip-components=1 bin/micromamba
    chmod +x "${MICROMAMBA}"
    echo "  ✓ Installed: $(${MICROMAMBA} --version)"
fi
echo ""

# ═════════════════════════════════════════════════════════════════
# STEP 2: Create conda env with Python 3.11 + glibc 2.28 sysroot + patchelf
#
# KEY PACKAGES:
#   - python=3.11        : The Python interpreter
#   - sysroot_linux-64   : Provides glibc 2.28 runtime (libc.so.6, ld-linux)
#   - patchelf           : Tool to modify ELF binaries' interpreter
#   - libstdcxx-ng>=12   : Modern C++ runtime (needed by torch/jax .so files)
#   - coreutils          : Provides uname, etc. linked to conda's libs
#
# We do NOT install CUDA here — PyTorch and JAX pip wheels bundle their own
# CUDA runtime. The GPU DRIVER on cluster nodes is what matters (must be >=525).
# ═════════════════════════════════════════════════════════════════
echo "[2/6] Creating conda environment with glibc 2.28 sysroot..."

if [[ -d "${VENV}" ]]; then
    echo "  ⚠ Removing old environment at ${VENV}..."
    rm -rf "${VENV}"
fi

echo "  → Installing Python 3.11 + sysroot (glibc 2.28) + patchelf + coreutils..."
echo "    (This takes 2-5 minutes)"
echo ""

${MICROMAMBA} create -p "${VENV}" \
    python=3.11 \
    pip \
    sysroot_linux-64=2.28 \
    patchelf \
    coreutils \
    "libstdcxx-ng>=12" \
    -c conda-forge \
    -y

echo ""
echo "  ✓ Conda env created at ${VENV}"
echo "  ✓ Python: $(${VENV}/bin/python --version)"
echo ""

# ═════════════════════════════════════════════════════════════════
# STEP 3: Patch Python to use glibc 2.28
#
# This is the CRITICAL step. We change Python's ELF interpreter from the
# system's ld-linux (glibc 2.17) to conda sysroot's ld-linux (glibc 2.28).
#
# We also set RPATH so Python finds glibc 2.28 libs WITHOUT LD_LIBRARY_PATH.
# This is important because LD_LIBRARY_PATH would break system binaries
# (uname, gcc, etc.) that pip spawns via subprocess.
# ═════════════════════════════════════════════════════════════════
echo "[3/6] Patching Python to use glibc 2.28..."

# Locate the sysroot's dynamic linker
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
NEW_LD="${SYSROOT}/lib64/ld-linux-x86-64.so.2"
PATCHELF="${VENV}/bin/patchelf"
PYTHON="${VENV}/bin/python3.11"

# Verify sysroot has glibc 2.28
if [[ ! -f "${NEW_LD}" ]]; then
    echo "  ✘ ERROR: Dynamic linker not found at ${NEW_LD}"
    echo "    The sysroot_linux-64 package may not have installed correctly."
    echo "    Try: ${MICROMAMBA} install -p ${VENV} sysroot_linux-64=2.28 -c conda-forge -y"
    exit 1
fi

# Check current interpreter
OLD_INTERP=$(${PATCHELF} --print-interpreter "${PYTHON}" 2>/dev/null || echo "unknown")
echo "  Current interpreter: ${OLD_INTERP}"
echo "  New interpreter:     ${NEW_LD}"

# Apply patch: change ELF interpreter
${PATCHELF} --set-interpreter "${NEW_LD}" "${PYTHON}"

# Set RPATH so Python finds sysroot's glibc WITHOUT needing LD_LIBRARY_PATH
# This is what makes pip detect glibc 2.28 without breaking system binaries
ORIG_RPATH=$(${PATCHELF} --print-rpath "${PYTHON}" 2>/dev/null || echo "")
NEW_RPATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib"
if [[ -n "${ORIG_RPATH}" ]]; then
    NEW_RPATH="${NEW_RPATH}:${ORIG_RPATH}"
fi
${PATCHELF} --set-rpath "${NEW_RPATH}" "${PYTHON}"

# Verify Python still works after patching
if ! ${PYTHON} -c "print('Python works after patchelf')" 2>/dev/null; then
    echo "  ✘ ERROR: Python broken after patchelf!"
    echo "    Attempting recovery..."
    ${PATCHELF} --set-interpreter "${OLD_INTERP}" "${PYTHON}"
    echo "    Reverted. The sysroot approach failed on this system."
    exit 1
fi

# Verify glibc version detection (this is what pip uses to decide wheel compat)
DETECTED_GLIBC=$(${PYTHON} -c "
import ctypes
try:
    libc = ctypes.CDLL('libc.so.6')
    gnu_get_libc_version = libc.gnu_get_libc_version
    gnu_get_libc_version.restype = ctypes.c_char_p
    print(gnu_get_libc_version().decode())
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "error")

echo "  ✓ Patched! Detected glibc: ${DETECTED_GLIBC}"

if [[ "${DETECTED_GLIBC}" != 2.2* && "${DETECTED_GLIBC}" != 2.3* ]]; then
    echo "  ✘ ERROR: Expected glibc >= 2.28, got ${DETECTED_GLIBC}"
    echo "    The patchelf + RPATH approach did not work."
    echo "    Python's ctypes still loads the system glibc 2.17."
    echo ""
    echo "    Debug info:"
    echo "    Interpreter: $(${PATCHELF} --print-interpreter ${PYTHON})"
    echo "    RPATH: $(${PATCHELF} --print-rpath ${PYTHON})"
    echo "    ldd output:"
    ldd "${PYTHON}" 2>&1 | grep libc || true
    exit 1
fi
echo ""

# ═════════════════════════════════════════════════════════════════
# STEP 4: Install OpenPI with ALL dependencies
#
# NOW that Python sees glibc 2.28 (via RPATH, NOT LD_LIBRARY_PATH),
# pip will happily download manylinux_2_28 wheels.
#
# IMPORTANT: We do NOT set LD_LIBRARY_PATH here!
# System binaries (uname, gcc, etc.) spawned by pip would crash if
# they found glibc 2.28's libc.so.6 in their library search path.
# The RPATH on Python is sufficient for pip's glibc detection.
# ═════════════════════════════════════════════════════════════════
echo "[4/6] Installing OpenPI + all dependencies..."
echo "  (This takes 10-20 minutes — downloading PyTorch ~2.5GB, JAX ~700MB, etc.)"
echo ""
echo "  NOTE: LD_LIBRARY_PATH is intentionally UNSET during installation."
echo "  Python detects glibc 2.28 via RPATH. System binaries remain unaffected."
echo ""

# Ensure conda's bin is in PATH (provides coreutils like uname if needed)
export PATH="${VENV}/bin:${PATH}"

cd "${OPENPI}"

# Step 4a: Pre-install ml-dtypes at the pinned version (from [tool.uv] override)
# This prevents version conflicts during resolution
echo "  [4a] Pre-installing ml-dtypes==0.4.1 (version override)..."
${VENV}/bin/pip install --no-cache-dir "ml-dtypes==0.4.1" 2>&1 | tail -5
echo ""

# Step 4b: Install PyTorch with CUDA 12.4 from PyTorch's own index
# (PyTorch wheels bundle their own CUDA runtime, so no system CUDA needed)
echo "  [4b] Installing PyTorch 2.7.1 + CUDA 12.4..."
${VENV}/bin/pip install --no-cache-dir \
    "torch==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -5
echo ""

# Step 4c: Install the full OpenPI project (editable) with ALL dependencies
# This respects pyproject.toml's version pins and installs the complete dep tree
echo "  [4c] Installing OpenPI (editable) with full dependency tree..."
echo "       This installs: jax, flax, orbax, lerobot, transformers, wandb, etc."
${VENV}/bin/pip install --no-cache-dir \
    -e . \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tee /tmp/openpi_install.log | tail -20

INSTALL_STATUS=${PIPESTATUS[0]:-$?}

# Step 4d: Handle potential rerun-sdk failure
# lerobot depends on rerun-sdk which needs X11/GL (not available on headless HPC).
# If the full install failed due to rerun-sdk, do a targeted fix:
if [[ ${INSTALL_STATUS} -ne 0 ]]; then
    echo ""
    echo "  ⚠ Full install had issues. Checking if it's rerun-sdk..."
    if grep -qi "rerun\|rerun-sdk\|rerun_sdk" /tmp/openpi_install.log; then
        echo "  → rerun-sdk failed (expected on headless HPC). Working around..."
        echo ""

        # Install OpenPI as editable (no deps, just the package itself)
        ${VENV}/bin/pip install --no-cache-dir -e . --no-deps 2>&1 | tail -3

        # Install all deps from pyproject.toml EXCEPT lerobot
        ${VENV}/bin/pip install --no-cache-dir \
            "augmax>=0.3.4" "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" \
            "flatbuffers>=24.3.25" "flax==0.10.2" "fsspec[gcs]>=2024.6.0" \
            "imageio>=2.36.1" "jax[cuda12]==0.5.3" "jaxtyping==0.2.36" \
            "ml-collections==1.0.0" "numpy>=1.22.4" "numpydantic>=1.6.6" \
            "opencv-python>=4.10.0.84" "orbax-checkpoint==0.11.13" \
            "pillow>=11.0.0" "sentencepiece>=0.2.0" "tqdm-loggable>=0.2" \
            "typing-extensions>=4.12.2" "tyro>=0.9.5" "wandb>=0.19.1" \
            "filelock>=3.16.1" "beartype==0.19.0" "treescope>=0.1.7" \
            "transformers==4.53.2" "rich>=14.0.0" "polars>=1.30.0" \
            --extra-index-url https://download.pytorch.org/whl/cu124 \
            2>&1 | tail -10

        # Install lerobot WITHOUT rerun-sdk
        ${VENV}/bin/pip install --no-cache-dir "lerobot>=0.4.0" --no-deps 2>&1 | tail -3
        # Then install lerobot's key runtime deps that we actually use
        ${VENV}/bin/pip install --no-cache-dir \
            "datasets>=2.19" "huggingface-hub>=0.23" "safetensors" \
            "draccus" "jsonlines" "pyarrow>=15.0" "torchvision" \
            --extra-index-url https://download.pytorch.org/whl/cu124 \
            2>&1 | tail -5

        echo "  ✓ Installed with rerun-sdk workaround"
    else
        echo "  ✘ Install failed for a different reason."
        echo "  Last 30 lines of /tmp/openpi_install.log:"
        tail -30 /tmp/openpi_install.log
        exit 1
    fi
fi
echo ""

# Step 4e: Install openpi-client package
echo "  [4e] Installing openpi-client..."
if [[ -d "${OPENPI}/packages/openpi-client" ]]; then
    ${VENV}/bin/pip install --no-cache-dir -e "${OPENPI}/packages/openpi-client" 2>&1 | tail -3
    echo "  ✓ openpi-client installed"
else
    echo "  ⚠ openpi-client directory not found (skipping)"
fi

echo ""
echo "  ✓ All packages installed!"
echo ""

# ═════════════════════════════════════════════════════════════════
# STEP 5: Setup directories, symlinks, env activation script
# ═════════════════════════════════════════════════════════════════
echo "[5/6] Setting up paths & activation script..."

# Dataset & logs directories
mkdir -p "${DATASET_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname ${HF_SYMLINK})"

# Create or update symlink
if [[ -L "${HF_SYMLINK}" ]]; then
    rm -f "${HF_SYMLINK}"
fi
ln -sf "${DATASET_DIR}" "${HF_SYMLINK}"

# Create the activation script for SLURM runtime
# THIS is where LD_LIBRARY_PATH gets set — only for runtime, never during install
cat > "${VENV}/activate_hpc.sh" << ACTIVATE_EOF
#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Source this in SLURM scripts BEFORE running Python:
#   source "${VENV}/activate_hpc.sh"
#
# This sets LD_LIBRARY_PATH so that dynamically loaded .so files
# (torch, jax, etc.) can find glibc 2.28 symbols at runtime.
#
# DO NOT source this before running pip/system commands!
# ═══════════════════════════════════════════════════════════════
export PATH="${VENV}/bin:\${PATH}"
export LD_LIBRARY_PATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib:\${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${VENV}"
export HF_HOME="${BEEGFS}/.cache/huggingface"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export WANDB_PROJECT="rlt-ur5e"
ACTIVATE_EOF
chmod +x "${VENV}/activate_hpc.sh"

echo "  ✓ Activation script: ${VENV}/activate_hpc.sh"
echo "  ✓ Dataset symlink: ${HF_SYMLINK} → ${DATASET_DIR}"
echo "  ✓ Log directory: ${LOG_DIR}"
echo ""

# ═════════════════════════════════════════════════════════════════
# STEP 6: Verify the installation
#
# NOW we source activate_hpc.sh (sets LD_LIBRARY_PATH) because we're
# done with pip and only running Python imports (no system binaries).
# ═════════════════════════════════════════════════════════════════
echo "[6/6] Verifying installation..."
echo ""

# Source activation (safe now — no more pip/subprocess calls to system binaries)
source "${VENV}/activate_hpc.sh"

cd "${OPENPI}"
${VENV}/bin/python << 'VERIFY_EOF'
import sys
import os

print("  Package Versions:")
print("  " + "─" * 50)

packages = [
    ("torch",          lambda: __import__("torch").__version__),
    ("jax",            lambda: __import__("jax").__version__),
    ("jaxlib",         lambda: __import__("jaxlib").__version__),
    ("flax",           lambda: __import__("flax").__version__),
    ("numpy",          lambda: __import__("numpy").__version__),
    ("orbax",          lambda: __import__("orbax.checkpoint").__version__),
    ("optax",          lambda: __import__("optax").__version__),
    ("wandb",          lambda: __import__("wandb").__version__),
    ("transformers",   lambda: __import__("transformers").__version__),
    ("lerobot",        lambda: __import__("lerobot").__version__),
    ("sentencepiece",  lambda: __import__("sentencepiece").__version__),
    ("polars",         lambda: __import__("polars").__version__),
]

all_ok = True
for name, get_ver in packages:
    try:
        ver = get_ver()
        print(f"  ✓ {name:16s} {ver}")
    except Exception as e:
        print(f"  ✘ {name:16s} FAILED: {e}")
        all_ok = False

print()
print("  OpenPI Import Chain (what train.py actually uses):")
print("  " + "─" * 50)

train_imports = [
    "openpi.models.model",
    "openpi.models.pi0_config",
    "openpi.models.pi0_fast",
    "openpi.training.checkpoints",
    "openpi.training.config",
    "openpi.training.data_loader",
    "openpi.training.optimizer",
    "openpi.training.sharding",
    "openpi.training.utils",
    "openpi.training.weight_loaders",
    "openpi.shared.download",
    "openpi.shared.normalize",
    "openpi.transforms",
    "openpi.policies.ur5e_policy",
]

for mod in train_imports:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except Exception as e:
        print(f"  ✘ {mod}: {e}")
        all_ok = False

print()
if all_ok:
    print("  🎉 ALL IMPORTS PASS! Environment is ready for training.")
else:
    print("  ⚠  Some imports failed. Check errors above.")
    sys.exit(1)
VERIFY_EOF

VERIFY_STATUS=$?
echo ""

if [[ ${VERIFY_STATUS} -eq 0 ]]; then
    echo "═════════════════════════════════════════════════════════════════"
    echo "  ✓ SETUP COMPLETE!"
    echo "═════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Environment: ${VENV}"
    echo "  Activate:    source ${VENV}/activate_hpc.sh"
    echo ""
    echo "  Next steps:"
    echo "    1. W&B login:  source ${VENV}/activate_hpc.sh && wandb login"
    echo "    2. Train:      cd ${PROJECT}/hpc && bash 03_train.sh both"
    echo "    3. Monitor:    bash 04_status.sh"
    echo ""
else
    echo "═════════════════════════════════════════════════════════════════"
    echo "  ⚠ SETUP NEEDS ATTENTION"
    echo "═════════════════════════════════════════════════════════════════"
    echo "  Some packages failed to import. Common fixes:"
    echo "    - Check /tmp/openpi_install.log for pip errors"
    echo "    - Try: source ${VENV}/activate_hpc.sh && python -c 'import torch'"
    echo "    - If glibc issues persist, check: ldd ${VENV}/bin/python3.11"
    exit 1
fi
