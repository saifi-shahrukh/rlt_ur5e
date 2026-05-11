#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Fix Python DT_RPATH so ALL shared libs in the process find sysroot glibc.
#
# KEY INSIGHT:
#   DT_RUNPATH (patchelf default) = only direct deps of python binary
#   DT_RPATH   (--force-rpath)    = propagates to ALL transitive deps!
#
# This means:
#   - jaxlib's .so loading librt → finds sysroot librt via DT_RPATH ✓
#   - multiprocessing workers (run python3.11) → have same RPATH ✓  
#   - ptxas (exec'd as separate binary) → NOT affected by RPATH ✓
#
# Run ONCE on HPC before submitting training jobs.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

VENV=/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"
PATCHELF="${VENV}/bin/patchelf"
PYTHON="${VENV}/bin/python3.11"

echo "═══════════════════════════════════════════════════════════════"
echo "  Fixing Python DT_RPATH (transitive) for all shared libs"
echo "═══════════════════════════════════════════════════════════════"

# Show current state
echo ""
echo "Before:"
echo "  Interpreter: $(${PATCHELF} --print-interpreter ${PYTHON})"
echo "  RPATH:       $(${PATCHELF} --print-rpath ${PYTHON})"
echo "  ELF type:"
readelf -d "${PYTHON}" 2>/dev/null | grep -E "RPATH|RUNPATH" || echo "    (none found)"

# The critical fix: --force-rpath converts DT_RUNPATH → DT_RPATH
NEW_RPATH="${SYSROOT}/lib64:${SYSROOT}/usr/lib64:${VENV}/lib"

echo ""
echo "Setting DT_RPATH (--force-rpath) to:"
echo "  ${NEW_RPATH}"
${PATCHELF} --force-rpath --set-rpath "${NEW_RPATH}" "${PYTHON}"

# Verify interpreter is sysroot
CURRENT_INTERP=$(${PATCHELF} --print-interpreter "${PYTHON}")
if [[ "${CURRENT_INTERP}" != *"sysroot"* ]]; then
    echo "Setting interpreter to sysroot ld-linux..."
    ${PATCHELF} --set-interpreter "${SYSROOT}/lib64/ld-linux-x86-64.so.2" "${PYTHON}"
fi

echo ""
echo "After:"
echo "  Interpreter: $(${PATCHELF} --print-interpreter ${PYTHON})"
echo "  RPATH:       $(${PATCHELF} --print-rpath ${PYTHON})"
echo "  ELF type:"
readelf -d "${PYTHON}" 2>/dev/null | grep -E "RPATH|RUNPATH" || echo "    (none found)"

# Test: the imports that were previously failing
echo ""
echo "Testing critical imports (no LD_LIBRARY_PATH):"
unset LD_LIBRARY_PATH
${PYTHON} -c "
import sys
print(f'  ✓ Python {sys.version}')

# This was failing with: /lib64/librt.so.1: undefined symbol: __clock_nanosleep
import jaxlib.cpu_feature_guard
print('  ✓ jaxlib.cpu_feature_guard (librt resolved correctly)')

# This was failing in workers
import multiprocessing.resource_tracker
print('  ✓ multiprocessing.resource_tracker (no __clock_nanosleep error)')

# Full JAX init
import jax
print(f'  ✓ JAX {jax.__version__} initialized')
print(f'  ✓ Devices: {jax.devices()}')

print()
print('  ALL CRITICAL IMPORTS OK — training will work!')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Done! DT_RPATH set. Submit training with:"
echo "    cd hpc && bash 03_train.sh 50demos"
echo "═══════════════════════════════════════════════════════════════"
