#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Submit training jobs to SLURM
# Usage:
#   bash 03_train.sh pi0              # Train π0 (9 demos)
#   bash 03_train.sh pi05             # Train π0.5 (9 demos)
#   bash 03_train.sh pi0_fast         # Train π0-FAST (9 demos)
#   bash 03_train.sh both             # Train π0 + π0.5 (9 demos)
#   bash 03_train.sh all              # Train all 3 models (9 demos)
#   bash 03_train.sh 50demos          # Train ALL 3 on 50-demo dataset
#   bash 03_train.sh pi0_50           # Train π0 (50 demos)
#   bash 03_train.sh pi05_50          # Train π0.5 (50 demos)
#   bash 03_train.sh pi0fast_50       # Train π0-FAST (50 demos)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
LOG_DIR="/data/beegfs/home/saifi/logs"

mkdir -p "${LOG_DIR}"

MODE="${1:-50demos}"

# Determine dataset
if [[ "${MODE}" == *"50"* ]]; then
    DEMOS="50 demos"
else
    DEMOS="9 demos"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  HPC Training — π0/π0.5/π0-FAST Peg Insertion (${DEMOS})"
echo "  GPU: V100 32GB | Optimized batch + workers"
echo "  W&B: wandb.ai/saifi/openpi"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check HF token
if [[ -f /data/beegfs/home/saifi/.cache/huggingface/token ]]; then
    echo "  ✓ HF Token: found (PaliGemma access)"
else
    echo "  ✗ HF Token: NOT FOUND — run: huggingface-cli login"
    exit 1
fi

# Check W&B
if grep -qs 'password' ~/.netrc 2>/dev/null; then
    echo "  ✓ W&B: API key found in ~/.netrc"
elif [[ -f "${HOME}/.config/wandb/api_key" ]]; then
    echo "  ✓ W&B: API key found"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "  ✓ W&B: API key found in env"
else
    echo "  ⚠ W&B: No API key — runs will log offline"
fi
echo ""

# Clean broken caches
rm -rf /data/beegfs/home/saifi/.cache/huggingface/datasets/parquet/default-*/*.incomplete 2>/dev/null || true

submit_job() {
    local name="$1"
    local script="$2"
    local job_id
    job_id=$(sbatch --parsable "${script}")
    echo "  ✓ ${name} → Job ${job_id}"
}

case "${MODE}" in
    # ─── 9-demo training ──────────────────────────────────────────────────
    pi0)
        echo "  → Submitting: π0 (9 demos)"
        submit_job "π0" "${SLURM_DIR}/pi0.sh"
        ;;
    pi05)
        echo "  → Submitting: π0.5 (9 demos)"
        submit_job "π0.5" "${SLURM_DIR}/pi05.sh"
        ;;
    pi0_fast|pi0fast|fast)
        echo "  → Submitting: π0-FAST (9 demos)"
        submit_job "π0-FAST" "${SLURM_DIR}/pi0_fast.sh"
        ;;
    both)
        echo "  �� Submitting: π0 + π0.5 (9 demos, staggered 30s)"
        submit_job "π0  " "${SLURM_DIR}/pi0.sh"
        echo "  ⏳ 30s..."; sleep 30
        submit_job "π0.5" "${SLURM_DIR}/pi05.sh"
        ;;
    all)
        echo "  → Submitting: ALL 3 models (9 demos, staggered 30s)"
        submit_job "π0    " "${SLURM_DIR}/pi0.sh"
        echo "  ⏳ 30s..."; sleep 30
        submit_job "π0.5  " "${SLURM_DIR}/pi05.sh"
        echo "  ⏳ 30s..."; sleep 30
        submit_job "π0-FAST" "${SLURM_DIR}/pi0_fast.sh"
        ;;
    # ─── 50-demo training (PRIMARY TARGET) ────────────────────────────────
    50demos|50)
        echo "  → Submitting: ALL 3 models (50 demos, staggered 30s)"
        echo "    Expected runtime: ~4-6 hours each on V100 32GB"
        echo ""
        submit_job "π0-FAST (50)" "${SLURM_DIR}/pi0_fast_50demos.sh"
        echo "  ⏳ 30s..."; sleep 30
        submit_job "π0 (50)    " "${SLURM_DIR}/pi0_50demos.sh"
        echo "  ⏳ 30s..."; sleep 30
        submit_job "π0.5 (50)  " "${SLURM_DIR}/pi05_50demos.sh"
        ;;
    pi0_50)
        echo "  → Submitting: π0 (50 demos)"
        submit_job "π0 (50)" "${SLURM_DIR}/pi0_50demos.sh"
        ;;
    pi05_50)
        echo "  → Submitting: π0.5 (50 demos)"
        submit_job "π0.5 (50)" "${SLURM_DIR}/pi05_50demos.sh"
        ;;
    pi0fast_50|pi0_fast_50)
        echo "  → Submitting: π0-FAST (50 demos)"
        submit_job "π0-FAST (50)" "${SLURM_DIR}/pi0_fast_50demos.sh"
        ;;
    *)
        echo "  Usage: bash 03_train.sh {pi0|pi05|pi0_fast|both|all|50demos|pi0_50|pi05_50|pi0fast_50}"
        exit 1
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor:"
echo "    squeue -u saifi"
echo "    tail -f ${LOG_DIR}/*.err"
echo "    W&B: https://wandb.ai/saifi/openpi"
echo "  Cancel all:"
echo "    scancel -u saifi"
echo "─────────────────────────────────────────────────────────────"
