#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Submit Training Jobs
# Run on: hpc-headnode.iis.fhg.de
# Usage: bash 03_train.sh [pi0|pi05|both]
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
DATASET="/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual"
LOG_DIR="/data/beegfs/home/saifi/logs"

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Training — π0/π0.5 Peg Insertion (9 demos)"
echo "  Norm stats: ✓ all 3 configs pre-computed in repo"
echo "  W&B: online (check wandb.ai for live curves)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Pre-flight ──────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

if [[ ! -f "${OPENPI}/.venv/bin/python" ]]; then
    echo "ERROR: Venv not found! Run: bash setup_hpc_env.sh"
    exit 1
fi

if [[ ! -f "${OPENPI}/.venv/activate_hpc.sh" ]]; then
    echo "ERROR: activate_hpc.sh not found! Run: bash setup_hpc_env.sh"
    exit 1
fi

if [[ -z "$(ls -A ${DATASET} 2>/dev/null)" ]]; then
    echo "⚠ WARNING: Dataset is empty at ${DATASET}"
    echo "  Run from LOCAL: bash 02_transfer_dataset.sh"
    read -p "  Continue anyway? [y/N] " -n 1 -r; echo ""
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Check W&B
if [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  ✓ W&B: configured (online logging)"
elif [[ -n "${WANDB_API_KEY}" ]]; then
    echo "  ✓ W&B: API key found in env"
else
    echo "  ⚠ W&B: not configured! Training will log offline."
    echo "    Fix: source ${OPENPI}/.venv/activate_hpc.sh && wandb login"
    echo ""
fi
echo ""

# ─── Parse argument ──────────────────────────────────────────────────────────
MODE="${1:-help}"

case "${MODE}" in
    pi0)
        echo "  → Submitting: π0 LoRA (30k steps, batch=16, ~2h)"
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi0.sh")
        echo "  ✓ Job ${JOB} submitted"
        echo "    Log: tail -f ${LOG_DIR}/pi0_peg_${JOB}.out"
        ;;

    pi05|pi0.5)
        echo "  → Submitting: π0.5 LoRA (30k steps, batch=16, ~3h)"
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi05.sh")
        echo "  ✓ Job ${JOB} submitted"
        echo "    Log: tail -f ${LOG_DIR}/pi05_peg_${JOB}.out"
        ;;

    both)
        echo "  → Submitting: π0 + π0.5 in parallel (separate GPU nodes)"
        echo ""
        J1=$(sbatch --parsable "${SLURM_DIR}/pi0.sh")
        echo "  ✓ π0   → Job ${J1} | Log: ${LOG_DIR}/pi0_peg_${J1}.out"
        J2=$(sbatch --parsable "${SLURM_DIR}/pi05.sh")
        echo "  ✓ π0.5 → Job ${J2} | Log: ${LOG_DIR}/pi05_peg_${J2}.out"
        echo ""
        echo "  Both running in parallel on different nodes."
        echo "  Watch on W&B: https://wandb.ai → project 'rlt-ur5e'"
        ;;

    *)
        echo "Usage: bash 03_train.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  pi0    Train π0 LoRA only (~2h on V100 32GB)"
        echo "  pi05   Train π0.5 LoRA only (~3h on V100 32GB)"
        echo "  both   Train π0 + π0.5 in parallel (RECOMMENDED)"
        echo ""
        echo "Examples:"
        echo "  bash 03_train.sh both   # train both models"
        echo "  bash 03_train.sh pi05   # just π0.5"
        echo ""
        echo "Note: Norm stats already in repo — no need to compute."
        exit 0
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor:"
echo "    bash 04_status.sh"
echo "    squeue -u saifi"
echo "    W&B: https://wandb.ai → project 'rlt-ur5e'"
echo "  Cancel:"
echo "    scancel <JOB_ID>"
echo "─────────────────────────────────────────────────────────────"
