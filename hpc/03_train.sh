#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Submit Training Jobs
# Run on: hpc-headnode.iis.fhg.de
# Usage: bash 03_train.sh [norm|pi0|pi05|both|all]
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
DATASET="/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual"
LOG_DIR="/data/beegfs/home/saifi/logs"

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Training — π0/π0.5 Peg Insertion"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Pre-flight ──────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

if [[ ! -f "${OPENPI}/.venv/bin/activate" ]]; then
    echo "ERROR: Venv not found! Run: bash 01_setup.sh"
    exit 1
fi

if [[ -z "$(ls -A ${DATASET} 2>/dev/null)" ]]; then
    echo "⚠ WARNING: Dataset is empty at ${DATASET}"
    echo "  Run: bash 02_transfer_dataset.sh (from LOCAL machine)"
    read -p "  Continue anyway? [y/N] " -n 1 -r; echo ""
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# ─── Parse argument ──────────────────────────────────────────────────────────
MODE="${1:-help}"

case "${MODE}" in
    norm|norm_stats)
        echo "  → Submitting: Normalization Stats (π0 + π0.5)"
        JOB=$(sbatch --parsable "${SLURM_DIR}/norm_stats.sh")
        echo "  ✓ Job ${JOB} submitted"
        echo "    Log: ${LOG_DIR}/norm_stats_${JOB}.out"
        ;;

    pi0)
        echo "  → Submitting: π0 LoRA (30k steps, batch=16)"
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi0.sh")
        echo "  ✓ Job ${JOB} submitted"
        echo "    Log: ${LOG_DIR}/pi0_peg_${JOB}.out"
        ;;

    pi05|pi0.5)
        echo "  → Submitting: π0.5 LoRA (30k steps, batch=16)"
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi05.sh")
        echo "  ✓ Job ${JOB} submitted"
        echo "    Log: ${LOG_DIR}/pi05_peg_${JOB}.out"
        ;;

    both)
        echo "  → Submitting: π0 + π0.5 in parallel"
        J1=$(sbatch --parsable "${SLURM_DIR}/pi0.sh")
        echo "  ✓ π0  → Job ${J1}"
        J2=$(sbatch --parsable "${SLURM_DIR}/pi05.sh")
        echo "  ✓ π0.5 → Job ${J2}"
        echo ""
        echo "  Both will run on separate nodes in parallel."
        ;;

    all)
        echo "  → Full pipeline: norm_stats → π0 + π0.5 (chained)"
        echo ""
        NORM=$(sbatch --parsable "${SLURM_DIR}/norm_stats.sh")
        echo "  ✓ Norm stats → Job ${NORM}"

        J1=$(sbatch --parsable --dependency=afterok:${NORM} "${SLURM_DIR}/pi0.sh")
        echo "  ✓ π0  → Job ${J1} (starts after norm stats)"

        J2=$(sbatch --parsable --dependency=afterok:${NORM} "${SLURM_DIR}/pi05.sh")
        echo "  ✓ π0.5 → Job ${J2} (starts after norm stats)"
        echo ""
        echo "  Pipeline: [${NORM}] norm_stats → [${J1}] π0"
        echo "                                 → [${J2}] π0.5"
        ;;

    *)
        echo "Usage: bash 03_train.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  norm       Compute normalization stats (required for π0.5)"
        echo "  pi0        Train π0 LoRA only (~2h on V100)"
        echo "  pi05       Train π0.5 LoRA only (~3h on V100)"
        echo "  both       Train π0 + π0.5 in parallel"
        echo "  all        norm_stats → then π0 + π0.5 (RECOMMENDED first time)"
        echo ""
        echo "Examples:"
        echo "  bash 03_train.sh all    # First time"
        echo "  bash 03_train.sh both   # Already have norm stats"
        echo "  bash 03_train.sh pi05   # Just π0.5"
        exit 0
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor: bash 04_status.sh"
echo "  Cancel:  scancel <JOB_ID>"
echo "  Queue:   squeue -u saifi"
echo "─────────────────────────────────────────────────────────────"
