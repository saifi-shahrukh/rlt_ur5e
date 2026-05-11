#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Fix Python RPATH so multiprocessing workers find sysroot libs
# without needing LD_LIBRARY_PATH (which breaks ptxas)
#
# Run ONCE on HPC headnode before submitting training jobs.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

VENV=/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
PATCHELF="${VENV}/bin/patchelf"
PYTHON="${VENV}/bin/python3.11"

echo "═══════════════════════════════════════════════════════════════"
echo "  Fixing Python RPATH for multiprocessing workers"
echo "═══════════════════════════════════════════════════════════════"

# Check current state
echo ""
echo "Current interpreter:"
${PATCHELF} --print-interpreter "${PYTHON}" 2>/dev/null || echo "  FAILED"
echo "Current RPATH:"
${PATCHELF} --print-rpath "${PYTHON}" 2>/dev/null || echo "  FAILED"

# Set RPATH to include sysroot libs
# This means workers (spawned via fork/exec of python3.11) will find
# sysroot glibc without needing LD_LIBRARY_PATH
NEW_RPATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib"

echo ""
echo "Setting RPATH to: ${NEW_RPATH}"
${PATCHELF} --set-rpath "${NEW_RPATH}" "${PYTHON}"

# Verify interpreter is sysroot (should already be from initial setup)
CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${PYTHON}")
if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
    echo "Setting interpreter to sysroot ld-linux..."
    ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" "${PYTHON}"
fi

echo ""
echo "After fix:"
echo "  Interpreter: $(${PATCHELF} --print-interpreter ${PYTHON})"
echo "  RPATH: $(${PATCHELF} --print-rpath ${PYTHON})"

# Test: python should work WITHOUT ld-linux wrapper and WITHOUT LD_LIBRARY_PATH
echo ""
echo "Testing python directly (no wrapper, no LD_LIBRARY_PATH):"
unset LD_LIBRARY_PATH
${PYTHON} -c "import sys; print(f'  Python {sys.version}'); import multiprocessing; print('  multiprocessing: OK')"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Done! Workers will now find sysroot libs via RPATH."
echo "  No LD_LIBRARY_PATH needed → ptxas works too!"
echo "═══════════════════════════════════════════════════════════════"
