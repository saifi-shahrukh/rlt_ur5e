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
# DT_RPATH propagates to ALL .so files loaded in the process
# This is what ld-linux's --library-path does, but baked into the binary
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

# Test 1: Basic python works
echo ""
echo "Test 1: Python import (no LD_LIBRARY_PATH, no wrapper):"
unset LD_LIBRARY_PATH
${PYTHON} -c "
import sys
print(f'  Python {sys.version}')
import ctypes
import ctypes.util
# This imports librt transitively
print('  ctypes: OK')
" || { echo "  FAILED!"; exit 1; }

# Test 2: jaxlib import (the one that was failing)
echo ""
echo "Test 2: Import jaxlib (needs librt via transitive deps):"
${PYTHON} -c "
try:
    import jaxlib.cpu_feature_guard
    print('  jaxlib.cpu_feature_guard: OK')
except ImportError as e:
    print(f'  FAILED: {e}')
    import sys; sys.exit(1)
" || { echo "  jaxlib import failed - fix didn't work!"; exit 1; }

# Test 3: multiprocessing (simulates workers)
echo ""
echo "Test 3: Multiprocessing spawn (simulates data workers):"
${PYTHON} -c "
import multiprocessing
multiprocessing.set_start_method('spawn')
import multiprocessing.resource_tracker
print('  multiprocessing resource_tracker: OK')
# Actually spawn a worker
from multiprocessing import Process, Queue
def worker(q):
    import sys
    q.put(f'Worker OK (Python {sys.version_info.major}.{sys.version_info.minor})')
q = Queue()
p = Process(target=worker, args=(q,))
p.start()
p.join(timeout=10)
if p.exitcode == 0:
    msg = q.get_nowait()
    print(f'  Spawned worker: {msg}')
else:
    print(f'  Worker FAILED with exit code {p.exitcode}')
    import sys; sys.exit(1)
" || { echo "  multiprocessing test failed!"; exit 1; }

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ ALL TESTS PASSED!"
echo "  - Python loads sysroot libs via DT_RPATH (transitive)"
echo "  - jaxlib finds librt from sysroot"
echo "  - Multiprocessing workers spawn correctly"
echo "  - ptxas (separate process) unaffected by RPATH"
echo "═══════════════════════════════════════════════════════════════"
