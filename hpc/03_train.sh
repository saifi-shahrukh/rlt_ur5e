#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Submit training jobs to SLURM
# Usage:
#   bash 03_train.sh pi0        # Train π0 only
#   bash 03_train.sh pi05       # Train π0.5 only
#   bash 03_train.sh pi0_fast   # Train π0-FAST only
#   bash 03_train.sh both       # Train π0 + π0.5 (staggered by 60s)
#   bash 03_train.sh all        # Train all 3 models (staggered)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
LOG_DIR="/data/beegfs/home/saifi/logs"

mkdir -p "${LOG_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Training — π0/π0.5/π0-FAST Peg Insertion (9 demos)"
echo "  Norm stats: ✓ all 3 configs pre-computed in repo"
echo "  W&B: logs to project 'openpi' at wandb.ai/saifi/openpi"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check HF token
if [[ -f /data/beegfs/home/saifi/.cache/huggingface/token ]]; then
    echo "  ✓ HF Token: found (PaliGemma access)"
else
    echo "  ✗ HF Token: NOT FOUND — run: huggingface-cli login"
    echo "    PaliGemma is gated: https://huggingface.co/google/paligemma-3b-pt-224"
    exit 1
fi

# Check W&B
if grep -qs 'password' ~/.netrc 2>/dev/null; then
    echo "  ✓ W&B: API key found in ~/.netrc"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "  ✓ W&B: API key found in env"
else
    echo "  ⚠ W&B: No API key found — runs will log offline"
    echo "    Fix: wandb login"
fi
echo ""

# Clean broken dataset caches from previous failures
rm -rf /data/beegfs/home/saifi/.cache/huggingface/datasets/parquet/default-*/*.incomplete 2>/dev/null || true

submit_job() {
    local name="$1"
    local script="$2"
    local job_id
    job_id=$(sbatch --parsable "${script}")
    echo "  ✓ ${name} → Job ${job_id} | Log: ${LOG_DIR}/$(grep -oP '(?<=--output=)[^ ]+' ${script} | sed "s/%j/${job_id}/")"
}

MODE="${1:-both}"

case "${MODE}" in
    pi0)
        echo "  → Submitting: π0 only"
        submit_job "π0" "${SLURM_DIR}/pi0.sh"
        ;;
    pi05)
        echo "  → Submitting: π0.5 only"
        submit_job "π0.5" "${SLURM_DIR}/pi05.sh"
        ;;
    pi0_fast|pi0fast|fast)
        echo "  → Submitting: π0-FAST only"
        submit_job "π0-FAST" "${SLURM_DIR}/pi0_fast.sh"
        ;;
    both)
        echo "  → Submitting: π0 + π0.5 (staggered 60s to avoid cache race)"
        echo ""
        submit_job "π0  " "${SLURM_DIR}/pi0.sh"
        echo "  ⏳ Waiting 60s before π0.5 (dataset cache needs time)..."
        sleep 60
        submit_job "π0.5" "${SLURM_DIR}/pi05.sh"
        ;;
    all)
        echo "  → Submitting: ALL 3 models (staggered 60s apart)"
        echo ""
        submit_job "π0    " "${SLURM_DIR}/pi0.sh"
        echo "  ⏳ Waiting 60s..."
        sleep 60
        submit_job "π0.5  " "${SLURM_DIR}/pi05.sh"
        echo "  ⏳ Waiting 60s..."
        sleep 60
        submit_job "π0-FAST" "${SLURM_DIR}/pi0_fast.sh"
        ;;
    *)
        echo "  Usage: bash 03_train.sh {pi0|pi05|pi0_fast|both|all}"
        exit 1
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor:"
echo "    squeue -u saifi"
echo "    tail -f ${LOG_DIR}/pi0_peg_<JOBID>.out"
echo "    W&B: https://wandb.ai/saifi/openpi"
echo "  Cancel:"
echo "    scancel <JOB_ID>"
echo "─────────────────────────────────────────────────────────────"
