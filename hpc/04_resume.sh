#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Resume training from step 4000 checkpoints
# Both π0 and π0-FAST were killed at 12hr limit but saved at step 4000
#
# Usage:
#   bash hpc/04_resume.sh           # Resume both models
#   bash hpc/04_resume.sh pi0       # Resume π0 only (~1 hour)
#   bash hpc/04_resume.sh pi0fast   # Resume π0-FAST only (~3 hours)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
LOG_DIR="/data/beegfs/home/saifi/logs"
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"

mkdir -p "${LOG_DIR}"

MODE="${1:-both}"

echo "═══════════════════════════════════════════════════════════════"
echo "  RESUME Training from Checkpoints"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check what checkpoints exist
check_checkpoint() {
    local config="$1"
    local dir="${OPENPI}/checkpoints/${config}/peg_insertion_50demos"
    if ls -d ${dir}/[0-9]* &>/dev/null; then
        local latest=$(ls -d ${dir}/[0-9]* | sort -V | tail -1)
        local step=$(basename ${latest})
        if [[ "${step}" == "4999" || "${step}" == "5000" ]]; then
            echo "  ✓ ${config}: COMPLETE (step ${step})"
            return 1  # already done
        else
            echo "  ⚠ ${config}: step ${step} (needs resume)"
            return 0  # needs resume
        fi
    else
        echo "  ✗ ${config}: NO checkpoint found!"
        return 1
    fi
}

echo "  Checking existing checkpoints..."
PI0_NEEDS_RESUME=false
PI0FAST_NEEDS_RESUME=false
PI05_NEEDS_RESUME=false

if check_checkpoint "pi0_ur5e_peg_insertion_lora"; then PI0_NEEDS_RESUME=true; fi
if check_checkpoint "pi0_fast_ur5e_peg_insertion_lora"; then PI0FAST_NEEDS_RESUME=true; fi
if check_checkpoint "pi05_ur5e_peg_insertion_lora"; then PI05_NEEDS_RESUME=true; fi
echo ""

submit_job() {
    local name="$1"
    local script="$2"
    local job_id
    job_id=$(sbatch --parsable "${script}")
    echo "  ✓ ${name} → Job ${job_id}"
}

case "${MODE}" in
    pi0)
        if [[ "${PI0_NEEDS_RESUME}" == "true" ]]; then
            echo "  → Resuming π0 (step 4000→5000, ~1.5 hours)"
            submit_job "π0 resume" "${SLURM_DIR}/pi0_50demos_resume.sh"
        else
            echo "  → π0 already complete or no checkpoint to resume from"
        fi
        ;;
    pi0fast|pi0_fast|fast)
        if [[ "${PI0FAST_NEEDS_RESUME}" == "true" ]]; then
            echo "  → Resuming π0-FAST (step 4000→5000, ~3 hours)"
            submit_job "π0-FAST resume" "${SLURM_DIR}/pi0_fast_50demos_resume.sh"
        else
            echo "  → π0-FAST already complete or no checkpoint to resume from"
        fi
        ;;
    pi05)
        if [[ "${PI05_NEEDS_RESUME}" == "true" ]]; then
            echo "  → Resuming π0.5"
            submit_job "π0.5 resume" "${SLURM_DIR}/pi05_50demos_resume.sh"
        else
            echo "  → π0.5 already complete or no checkpoint to resume from"
        fi
        ;;
    both|all)
        echo "  → Resuming models that need it:"
        echo ""
        if [[ "${PI0_NEEDS_RESUME}" == "true" ]]; then
            echo "  π0: ~1000 steps remaining (~2.5 hr at 9.3s/step)"
            submit_job "π0 resume     " "${SLURM_DIR}/pi0_50demos_resume.sh"
        fi
        if [[ "${PI0FAST_NEEDS_RESUME}" == "true" ]]; then
            echo "  π0-FAST: ~960 steps remaining (~2.8 hr at 10.6s/step)"
            submit_job "π0-FAST resume" "${SLURM_DIR}/pi0_fast_50demos_resume.sh"
        fi
        if [[ "${PI05_NEEDS_RESUME}" == "true" ]]; then
            submit_job "π0.5 resume   " "${SLURM_DIR}/pi05_50demos_resume.sh"
        fi
        if [[ "${PI0_NEEDS_RESUME}" == "false" && "${PI0FAST_NEEDS_RESUME}" == "false" && "${PI05_NEEDS_RESUME}" == "false" ]]; then
            echo "  All models already complete! Proceed to:"
            echo "    bash hpc/05_download_checkpoints.sh"
        fi
        ;;
    *)
        echo "  Usage: bash hpc/04_resume.sh {pi0|pi0fast|pi05|both|all}"
        exit 1
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor:"
echo "    squeue -u saifi"
echo "    tail -f ${LOG_DIR}/pi0_resume_*.err"
echo "    tail -f ${LOG_DIR}/pi0fast_resume_*.err"
echo "    W&B: https://wandb.ai/saifi/openpi"
echo ""
echo "  After ALL complete:"
echo "    bash hpc/05_download_checkpoints.sh"
echo "─────────────────────────────────────────────────────────────"
