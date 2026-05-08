#!/bin/bash
# ══════════════════════════════════════════════════════════════
# COMPLETE FIX: Install ALL missing lerobot + datasets deps
# Then test LOCALLY before submitting to SLURM
#
# Run on HPC headnode:
#   cd /data/beegfs/home/saifi/rlt_ur5e/hpc && bash fix_all_deps.sh
# ══════════════════════════════════════════════════════════════
set -e

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy
VENV="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv"
UV="${HOME}/.local/bin/uv"
PROJECT="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"

echo "══════════════════════════════════════════════════════════════"
echo "  Installing ALL missing dependencies..."
echo "══════════════════════════════════════════════════════════════"

# lerobot deps (it needs these - we installed lerobot with --no-deps)
${UV} pip install --python "${VENV}/bin/python" \
    "accelerate" \
    "diffusers" \
    "gymnasium" \
    "av" \
    "deepdiff" \
    "termcolor" \
    "packaging"

# datasets deps (installed with --no-deps)
${UV} pip install --python "${VENV}/bin/python" \
    "pyarrow-hotfix"

# Any other pure-python deps that might still be missing
${UV} pip install --python "${VENV}/bin/python" \
    "wadler_lindig" \
    2>/dev/null || echo "  (wadler_lindig not on PyPI - equinox not needed anyway)"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Testing LOCALLY (no SLURM) — must pass before submitting"
echo "══════════════════════════════════════════════════════════════"

cd "${PROJECT}"

# Keep testing until this passes
while true; do
    OUTPUT=$(${VENV}/bin/python -c "
import sys, os
os.chdir('${PROJECT}')
exec(open('scripts/train.py').read().split('def ')[0])
print('✓ All top-level imports pass')
" 2>&1)

    echo "$OUTPUT" | tail -5

    if echo "$OUTPUT" | grep -q '✓ All top-level imports pass'; then
        echo ""
        echo "🎉 ALL IMPORTS PASS!"
        break
    fi

    # Extract the missing module name
    MISSING=$(echo "$OUTPUT" | grep "ModuleNotFoundError" | sed "s/.*No module named '\([^']*\)'.*/\1/" | head -1)

    if [ -z "$MISSING" ]; then
        echo "❌ Unknown error (not a missing module). Full output:"
        echo "$OUTPUT"
        exit 1
    fi

    echo ""
    echo "  → Missing: $MISSING — installing..."
    ${UV} pip install --python "${VENV}/bin/python" "$MISSING" || {
        echo "  ⚠ Failed to install '$MISSING' directly, trying with underscore/dash variants..."
        DASH_NAME=$(echo "$MISSING" | tr '_' '-')
        ${UV} pip install --python "${VENV}/bin/python" "$DASH_NAME" || {
            echo "  ❌ Cannot install '$MISSING'. Manual fix needed."
            exit 1
        }
    }
    echo ""
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Submitting training jobs..."
echo "══════════════════════════════════════════════════════════════"

cd /data/beegfs/home/saifi/rlt_ur5e/hpc
bash 03_train.sh both

echo ""
echo "  Waiting 120s then checking status..."
sleep 120

echo ""
squeue -u saifi

# Check for errors
for f in /data/beegfs/home/saifi/logs/pi0_peg_*.err /data/beegfs/home/saifi/logs/pi05_peg_*.err; do
    if [ -f "$f" ]; then
        size=$(stat --format=%s "$f" 2>/dev/null || echo 0)
        if [ "$size" -gt 0 ]; then
            echo "  ⚠ Error in $(basename $f):"
            tail -3 "$f"
        fi
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE. If no errors above, training is running!"
echo "  Check: https://wandb.ai → project 'rlt-ur5e'"
echo "══════════════════════════════════════════════════════════════"
