#!/bin/bash
# ══════════════════════════════════════════════════════════════
# DEFINITIVE FIX: Install ALL missing deps + resubmit training
# Run this ONCE on HPC headnode:
#   cd /data/beegfs/home/saifi/rlt_ur5e/hpc && bash fix_and_run.sh
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

# The ACTUAL error: gemma_pytorch.py imports pytest for type hints
${UV} pip install --python "${VENV}/bin/python" "pytest"

# Additional deps that might be missing (all pure-python, safe)
${UV} pip install --python "${VENV}/bin/python" \
    "wadler_lindig" \
    "pytest" \
    2>/dev/null || true

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Verifying imports (what train.py ACTUALLY needs)..."
echo "══════════════════════════════════════════════════════════════"

cd "${PROJECT}"

# Test EXACTLY what train.py imports — no more, no less
${VENV}/bin/python -c "
import sys, os
os.chdir('${PROJECT}')

print('Testing actual train.py import chain...')

# Direct train.py imports
import etils.epath; print('  ✓ etils.epath')
import flax.nnx; print('  ✓ flax.nnx')
import flax.training.common_utils; print('  ✓ flax.training.common_utils')
import flax.traverse_util; print('  ✓ flax.traverse_util')
import jax; print('  ✓ jax')
import jax.numpy; print('  ✓ jax.numpy')
import numpy; print('  ✓ numpy')
import optax; print('  ✓ optax')
import tqdm_loggable.auto; print('  ✓ tqdm_loggable')
import wandb; print('  ✓ wandb')

# openpi imports (follow the chain)
import openpi.models.model; print('  ✓ openpi.models.model')
import openpi.training.checkpoints; print('  ✓ openpi.training.checkpoints')
import openpi.training.config; print('  ✓ openpi.training.config')
import openpi.training.data_loader; print('  ✓ openpi.training.data_loader')
import openpi.training.optimizer; print('  ✓ openpi.training.optimizer')
import openpi.training.sharding; print('  ✓ openpi.training.sharding')
import openpi.training.utils; print('  ✓ openpi.training.utils')
import openpi.training.weight_loaders; print('  ✓ openpi.training.weight_loaders')
import openpi.shared.download; print('  ✓ openpi.shared.download')
import openpi.shared.normalize; print('  ✓ openpi.shared.normalize')
import openpi.transforms; print('  ✓ openpi.transforms')
import openpi.policies.ur5e_policy; print('  ✓ openpi.policies.ur5e_policy')

print()
print('🎉 ALL TRAIN.PY IMPORTS PASS!')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test FAILED. Check error above."
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Submitting training jobs..."
echo "══════════════════════════════════════════════════════════════"

cd /data/beegfs/home/saifi/rlt_ur5e/hpc
bash 03_train.sh both

echo ""
echo "Waiting 120s for jobs to start..."
sleep 120

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Checking job status..."
echo "══════════════════════════════════════════════════════════════"
squeue -u saifi

echo ""
echo "  Checking for errors..."
for f in /data/beegfs/home/saifi/logs/pi0_peg_*.err /data/beegfs/home/saifi/logs/pi05_peg_*.err; do
    if [ -f "$f" ]; then
        size=$(stat --format=%s "$f" 2>/dev/null || echo 0)
        if [ "$size" -gt 0 ]; then
            echo "  ⚠ Error in $(basename $f):"
            tail -5 "$f"
            echo ""
        fi
    fi
done

echo ""
echo "  Checking latest output..."
for f in /data/beegfs/home/saifi/logs/pi0_peg_*.out /data/beegfs/home/saifi/logs/pi05_peg_*.out; do
    if [ -f "$f" ]; then
        size=$(stat --format=%s "$f" 2>/dev/null || echo 0)
        if [ "$size" -gt 100 ]; then
            echo "  ── $(basename $f) (last 5 lines):"
            tail -5 "$f"
            echo ""
        fi
    fi
done

echo "══════════════════════════════════════════════════════════════"
echo "  DONE. Check W&B: https://wandb.ai → project 'rlt-ur5e'"
echo "══════════════════════════════════════════════════════════════"
