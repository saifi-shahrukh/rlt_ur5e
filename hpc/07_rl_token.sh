#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# RL Token Pipeline on HPC
# Extracts VLM embeddings + trains RL Token for all 3 models
#
# Usage:
#   bash hpc/07_rl_token.sh              # All 3 models
#   bash hpc/07_rl_token.sh pi0fast      # pi0-FAST only
#   bash hpc/07_rl_token.sh pi0          # pi0 only
#   bash hpc/07_rl_token.sh pi05         # pi0.5 only
#   bash hpc/07_rl_token.sh extract      # Extract only (all 3)
#   bash hpc/07_rl_token.sh train        # Train only (all 3, needs embeddings)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
LOG_DIR="/data/beegfs/home/saifi/logs"
OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
RLT="/data/beegfs/home/saifi/rlt_ur5e"
OUTPUT_DIR="${RLT}/checkpoints/rl_token"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

MODE="${1:-all}"

echo "═══════════════════════════════════════════════════════════════"
echo "  RL Token Pipeline (HPC GPU)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check which models have completed training
check_vla_ready() {
    local config="$1"
    local dir="${OPENPI}/checkpoints/${config}/peg_insertion_50demos"
    local latest=$(ls -d ${dir}/[0-9]* 2>/dev/null | sort -V | tail -1)
    if [[ -n "${latest}" ]]; then
        local step=$(basename ${latest})
        echo "  ✓ ${config}: step ${step}"
        return 0
    else
        echo "  ✗ ${config}: NO checkpoint found"
        return 1
    fi
}

check_embeddings() {
    local config="$1"
    local step="$2"
    local file="${OUTPUT_DIR}/embeddings_${config}_step${step}.pt"
    if [[ -f "${file}" ]]; then
        echo "  ✓ Embeddings: $(du -h ${file} | cut -f1)"
        return 0
    else
        echo "  ✗ Embeddings not yet extracted"
        return 1
    fi
}

echo "  VLA Training Status:"
PI0_READY=false; PI0FAST_READY=false; PI05_READY=false
PI0_STEP=""; PI0FAST_STEP=""; PI05_STEP=""

if check_vla_ready "pi0_ur5e_peg_insertion_lora"; then
    PI0_READY=true
    PI0_STEP=$(ls -d ${OPENPI}/checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/[0-9]* 2>/dev/null | sort -V | tail -1 | xargs basename)
fi
if check_vla_ready "pi0_fast_ur5e_peg_insertion_lora"; then
    PI0FAST_READY=true
    PI0FAST_STEP=$(ls -d ${OPENPI}/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/[0-9]* 2>/dev/null | sort -V | tail -1 | xargs basename)
fi
if check_vla_ready "pi05_ur5e_peg_insertion_lora"; then
    PI05_READY=true
    PI05_STEP=$(ls -d ${OPENPI}/checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/[0-9]* 2>/dev/null | sort -V | tail -1 | xargs basename)
fi
echo ""

submit_extract() {
    local config="$1"
    local step="$2"
    local job_id
    job_id=$(VLA_CONFIG="${config}" VLA_STEP="${step}" sbatch --parsable --export=ALL,VLA_CONFIG="${config}",VLA_STEP="${step}" "${SLURM_DIR}/extract_embeddings.sh")
    echo "  ✓ Extract ${config} → Job ${job_id}" >&2
    echo "${job_id}"
}

submit_train() {
    local config="$1"
    local step="$2"
    local dependency="$3"
    local dep_flag=""
    [[ -n "${dependency}" ]] && dep_flag="--dependency=afterok:${dependency}"
    local job_id
    job_id=$(sbatch --parsable ${dep_flag} --export=ALL,VLA_CONFIG="${config}",VLA_STEP="${step}" "${SLURM_DIR}/train_rl_token.sh")
    echo "  ✓ Train RL Token ${config} → Job ${job_id}"
}

case "${MODE}" in
    pi0fast|pi0_fast|fast)
        if [[ "${PI0FAST_READY}" == "true" ]]; then
            echo "  → pi0-FAST (step ${PI0FAST_STEP})"
            EXT_JOB=$(submit_extract "pi0_fast_ur5e_peg_insertion_lora" "${PI0FAST_STEP}")
            submit_train "pi0_fast_ur5e_peg_insertion_lora" "${PI0FAST_STEP}" "${EXT_JOB}"
        else
            echo "  ✗ pi0-FAST not ready"
        fi
        ;;
    pi0)
        if [[ "${PI0_READY}" == "true" ]]; then
            echo "  → pi0 (step ${PI0_STEP})"
            EXT_JOB=$(submit_extract "pi0_ur5e_peg_insertion_lora" "${PI0_STEP}")
            submit_train "pi0_ur5e_peg_insertion_lora" "${PI0_STEP}" "${EXT_JOB}"
        else
            echo "  ✗ pi0 not ready"
        fi
        ;;
    pi05)
        if [[ "${PI05_READY}" == "true" ]]; then
            echo "  → pi0.5 (step ${PI05_STEP})"
            EXT_JOB=$(submit_extract "pi05_ur5e_peg_insertion_lora" "${PI05_STEP}")
            submit_train "pi05_ur5e_peg_insertion_lora" "${PI05_STEP}" "${EXT_JOB}"
        else
            echo "  ✗ pi0.5 not ready"
        fi
        ;;
    extract)
        echo "  → Extracting embeddings for all ready models"
        [[ "${PI0_READY}" == "true" ]] && submit_extract "pi0_ur5e_peg_insertion_lora" "${PI0_STEP}"
        [[ "${PI0FAST_READY}" == "true" ]] && submit_extract "pi0_fast_ur5e_peg_insertion_lora" "${PI0FAST_STEP}"
        [[ "${PI05_READY}" == "true" ]] && submit_extract "pi05_ur5e_peg_insertion_lora" "${PI05_STEP}"
        ;;
    train)
        echo "  → Training RL Token for all models (needs embeddings already extracted)"
        [[ "${PI0_READY}" == "true" ]] && submit_train "pi0_ur5e_peg_insertion_lora" "${PI0_STEP}" ""
        [[ "${PI0FAST_READY}" == "true" ]] && submit_train "pi0_fast_ur5e_peg_insertion_lora" "${PI0FAST_STEP}" ""
        [[ "${PI05_READY}" == "true" ]] && submit_train "pi05_ur5e_peg_insertion_lora" "${PI05_STEP}" ""
        ;;
    all)
        echo "  → Full pipeline for all ready models (extract → train)"
        echo "    Jobs will chain: extract completes → train starts"
        echo ""
        if [[ "${PI0FAST_READY}" == "true" ]]; then
            echo "  pi0-FAST (step ${PI0FAST_STEP}):"
            EXT_JOB=$(submit_extract "pi0_fast_ur5e_peg_insertion_lora" "${PI0FAST_STEP}")
            submit_train "pi0_fast_ur5e_peg_insertion_lora" "${PI0FAST_STEP}" "${EXT_JOB}"
            echo ""
        fi
        if [[ "${PI0_READY}" == "true" ]]; then
            echo "  pi0 (step ${PI0_STEP}):"
            EXT_JOB=$(submit_extract "pi0_ur5e_peg_insertion_lora" "${PI0_STEP}")
            submit_train "pi0_ur5e_peg_insertion_lora" "${PI0_STEP}" "${EXT_JOB}"
            echo ""
        fi
        if [[ "${PI05_READY}" == "true" ]]; then
            echo "  pi0.5 (step ${PI05_STEP}):"
            EXT_JOB=$(submit_extract "pi05_ur5e_peg_insertion_lora" "${PI05_STEP}")
            submit_train "pi05_ur5e_peg_insertion_lora" "${PI05_STEP}" "${EXT_JOB}"
            echo ""
        fi
        ;;
    *)
        echo "  Usage: bash hpc/07_rl_token.sh {all|pi0|pi0fast|pi05|extract|train}"
        exit 1
        ;;
esac

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor:"
echo "    squeue -u saifi"
echo "    tail -f ${LOG_DIR}/extract_emb_*.err"
echo "    tail -f ${LOG_DIR}/rl_token_*.err"
echo ""
echo "  Output files:"
echo "    ${OUTPUT_DIR}/embeddings_*.pt  (VLM embeddings)"
echo "    ${OUTPUT_DIR}/*_rl_token.pt    (trained RL Token models)"
echo ""
echo "  After completion, download:"
echo "    scp saifi@hpc-headnode.iis.fhg.de:${OUTPUT_DIR}/*_rl_token.pt ./checkpoints/rl_token/"
echo "─────────────────────────────────────────────────────────────"
