#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Get Interactive GPU Session
# Run on: hpc-headnode.iis.fhg.de
# Usage: bash 06_interactive.sh [GPUS] [HOURS]
# ═══════════════════════════════════════════════════════════════════════════════

GPUS="${1:-1}"
HOURS="${2:-4}"
MEM="${3:-64G}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Interactive GPU Session"
echo "  GPUs: ${GPUS}× V100 32GB | Time: ${HOURS}h | Mem: ${MEM}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Once inside:"
echo "    cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
echo "    source .venv/activate_hpc.sh"
echo "    nvidia-smi"
echo "    python -c \"import jax; print(jax.devices())\""
echo ""
echo "  Exit with: Ctrl+D or 'exit'"
echo "─────────────────────────────────────────────────────────────"
echo ""

srun \
    --partition=gpu \
    --gres=gpu:${GPUS} \
    --cpus-per-task=8 \
    --mem=${MEM} \
    --time=${HOURS}:00:00 \
    --pty bash
