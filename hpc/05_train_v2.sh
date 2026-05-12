#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Dispatcher: Train v2 models (2-camera, proper LoRA ranks)
#
# Usage:
#   bash hpc/05_train_v2.sh all        # Submit all 3 models
#   bash hpc/05_train_v2.sh pi0        # Submit pi0 v2 only
#   bash hpc/05_train_v2.sh pi0fast    # Submit pi0-FAST v2 only
#   bash hpc/05_train_v2.sh pi05       # Submit pi0.5 v2 only
#
# v2 improvements over v1:
#   - 2 images only (no duplicate wrist) → ~25% faster training
#   - pi0-FAST uses rank=16 (not rank=4)
#   - Proper config names: *_v2_lora
#   - batch_size tuned for 2-image workload
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"

MODE="${1:-all}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Train v2 Models (2-camera, standard LoRA)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Mode: ${MODE}"
echo ""

case "${MODE}" in
    all)
        echo "  Submitting ALL 3 v2 models..."
        JOB1=$(sbatch --parsable "${SLURM_DIR}/pi0_v2_50demos.sh")
        echo "  ✓ pi0 v2     → Job ${JOB1}"
        JOB2=$(sbatch --parsable "${SLURM_DIR}/pi0_fast_v2_50demos.sh")
        echo "  ✓ pi0-FAST v2 → Job ${JOB2}"
        JOB3=$(sbatch --parsable "${SLURM_DIR}/pi05_v2_50demos.sh")
        echo "  ✓ pi0.5 v2   → Job ${JOB3}"
        ;;
    pi0)
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi0_v2_50demos.sh")
        echo "  ✓ pi0 v2     → Job ${JOB}"
        ;;
    pi0fast|pi0_fast|fast)
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi0_fast_v2_50demos.sh")
        echo "  ✓ pi0-FAST v2 → Job ${JOB}"
        ;;
    pi05)
        JOB=$(sbatch --parsable "${SLURM_DIR}/pi05_v2_50demos.sh")
        echo "  ✓ pi0.5 v2   → Job ${JOB}"
        ;;
    *)
        echo "  ERROR: Unknown mode '${MODE}'"
        echo "  Usage: bash hpc/05_train_v2.sh [all|pi0|pi0fast|pi05]"
        exit 1
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor:"
echo "    squeue -u saifi"
echo "    tail -f /data/beegfs/home/saifi/logs/pi0_v2_*.err"
echo "    tail -f /data/beegfs/home/saifi/logs/pi0_fast_v2_*.err"
echo "    tail -f /data/beegfs/home/saifi/logs/pi05_v2_*.err"
echo ""
echo "  Resume (after 12hr kill):"
echo "    Change --overwrite to --resume in the SLURM script, then resubmit"
echo "─────────────────────────────────────────────────────────────"
