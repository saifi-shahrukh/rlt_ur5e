#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Check Training Status
# Run on: hpc-headnode.iis.fhg.de
# ═══════════════════════════════════════════════════════════════════════════════

LOG_DIR="/data/beegfs/home/saifi/logs"
CHECKPOINT_DIR="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints"

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Status — $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Your Jobs ───────────────────────────────────────────────────────────────
echo "┌─── Your Jobs ──────────────────────────────────────────────┐"
JOBS=$(squeue -u saifi 2>/dev/null)
if [[ $(echo "${JOBS}" | wc -l) -le 1 ]]; then
    echo "  No jobs running or pending."
else
    echo "${JOBS}"
fi
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# ─── GPU Availability ────────────────────────────────────────────────────────
echo "┌─── GPU Nodes ────────────────────────────────────────────────┐"
sinfo -p gpu -O NodeList:20,Gres:16,GresUsed:22,StateLong:10 2>/dev/null || echo "  (sinfo unavailable)"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# ─── Latest Log ──────────────────────────────────────────────────────────────
if [[ -d "${LOG_DIR}" ]]; then
    LATEST=$(ls -t ${LOG_DIR}/*.out 2>/dev/null | head -1)
    if [[ -n "${LATEST}" ]]; then
        echo "┌─── Latest Log: $(basename ${LATEST}) ─────────────────────┐"
        echo "  ..."
        tail -15 "${LATEST}" | sed 's/^/  /'
        echo "└──────────────────────────────────────────────────────────────┘"
        echo ""
    fi

    # Show errors if any
    LATEST_ERR=$(ls -t ${LOG_DIR}/*.err 2>/dev/null | head -1)
    if [[ -n "${LATEST_ERR}" && -s "${LATEST_ERR}" ]]; then
        echo "┌─── Latest Errors: $(basename ${LATEST_ERR}) ──────────────┐"
        tail -10 "${LATEST_ERR}" | sed 's/^/  /'
        echo "└──────────────────────────────────────────────────────────────┘"
        echo ""
    fi
fi

# ─── Checkpoints ─────────────────────────────────────────────────────────────
if [[ -d "${CHECKPOINT_DIR}" ]]; then
    echo "┌─── Checkpoints ──────────────────────────────────────────┐"
    find "${CHECKPOINT_DIR}" -maxdepth 3 -type d | sort | sed 's/^/  /'
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""
fi

# ─── W&B ─────────────────────────────────────────────────────────────────────
echo "┌─── W&B ────────────────────────────────────────────────────────┐"
if [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  ✓ W&B configured — check https://wandb.ai for live plots"
else
    echo "  ⚠ W&B not configured (logging offline)"
    echo "    To enable: source .venv/bin/activate && wandb login"
fi
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# ─── Disk ────────────────────────────────────────────────────────────────────
echo "┌─── Disk Usage ─────────────────────────────────────────────┐"
echo "  BeeGFS:     $(du -sh /data/beegfs/home/saifi/ 2>/dev/null | cut -f1)"
echo "  Dataset:    $(du -sh /data/beegfs/home/saifi/datasets/ 2>/dev/null | cut -f1)"
[[ -d "${CHECKPOINT_DIR}" ]] && echo "  Checkpoints: $(du -sh ${CHECKPOINT_DIR} 2>/dev/null | cut -f1)"
echo "└──────────────────────────────────────────────────────────────┘"
