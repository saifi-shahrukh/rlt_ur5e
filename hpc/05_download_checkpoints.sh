#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Download Checkpoints from HPC
# Run on: LOCAL machine (robot workstation)
#
# After all 3 models complete training, download checkpoints for inference.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Download Trained Checkpoints ← HPC"
echo "═══════════════════════════════════════════════════════════════"
echo ""

HPC="saifi@hpc-headnode.iis.fhg.de"
HPC_CKPT="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints"

# Find local destination
LOCAL_PATHS=(
    "${HOME}/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints"
    "/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints"
)

LOCAL_DEST=""
for p in "${LOCAL_PATHS[@]}"; do
    if [[ -d "$(dirname $p)" ]]; then
        LOCAL_DEST="$p"
        break
    fi
done
[[ -z "${LOCAL_DEST}" ]] && LOCAL_DEST="${HOME}/hpc_checkpoints"
mkdir -p "${LOCAL_DEST}"

# Show what's available
echo "  Checking HPC checkpoints..."
echo ""
ssh "${HPC}" "echo '  Available checkpoints:' && \
    for d in ${HPC_CKPT}/*/peg_insertion_50demos/; do \
        if [ -d \"\$d\" ]; then \
            config=\$(basename \$(dirname \$d)); \
            steps=\$(ls -d \${d}[0-9]* 2>/dev/null | sort -V | tail -1 | xargs basename 2>/dev/null); \
            echo \"    ✓ \${config}: step \${steps:-none}\"; \
        fi; \
    done" || echo "  (connection failed - run from local machine)"
echo ""

echo "  Download options:"
echo "    1) All 3 models (π0 + π0.5 + π0-FAST)"
echo "    2) π0 only"
echo "    3) π0.5 only"
echo "    4) π0-FAST only"
echo "    5) Only params/ dirs (smaller, inference only)"
read -p "  Choice [1-5]: " CHOICE
echo ""

case "${CHOICE}" in
    1)
        echo "  Downloading all 3 models (full checkpoints)..."
        rsync -avz --progress \
            "${HPC}:${HPC_CKPT}/pi0_ur5e_peg_insertion_lora/" \
            "${LOCAL_DEST}/pi0_ur5e_peg_insertion_lora/"
        rsync -avz --progress \
            "${HPC}:${HPC_CKPT}/pi05_ur5e_peg_insertion_lora/" \
            "${LOCAL_DEST}/pi05_ur5e_peg_insertion_lora/"
        rsync -avz --progress \
            "${HPC}:${HPC_CKPT}/pi0_fast_ur5e_peg_insertion_lora/" \
            "${LOCAL_DEST}/pi0_fast_ur5e_peg_insertion_lora/"
        ;;
    2)
        rsync -avz --progress \
            "${HPC}:${HPC_CKPT}/pi0_ur5e_peg_insertion_lora/" \
            "${LOCAL_DEST}/pi0_ur5e_peg_insertion_lora/"
        ;;
    3)
        rsync -avz --progress \
            "${HPC}:${HPC_CKPT}/pi05_ur5e_peg_insertion_lora/" \
            "${LOCAL_DEST}/pi05_ur5e_peg_insertion_lora/"
        ;;
    4)
        rsync -avz --progress \
            "${HPC}:${HPC_CKPT}/pi0_fast_ur5e_peg_insertion_lora/" \
            "${LOCAL_DEST}/pi0_fast_ur5e_peg_insertion_lora/"
        ;;
    5)
        echo "  Downloading params only (for inference)..."
        for config in pi0_ur5e_peg_insertion_lora pi05_ur5e_peg_insertion_lora pi0_fast_ur5e_peg_insertion_lora; do
            echo "  → ${config}"
            # Get latest step
            LATEST=$(ssh "${HPC}" "ls -d ${HPC_CKPT}/${config}/peg_insertion_50demos/[0-9]* 2>/dev/null | sort -V | tail -1")
            if [[ -n "${LATEST}" ]]; then
                STEP=$(basename ${LATEST})
                mkdir -p "${LOCAL_DEST}/${config}/peg_insertion_50demos/${STEP}"
                rsync -avz --progress \
                    "${HPC}:${LATEST}/params/" \
                    "${LOCAL_DEST}/${config}/peg_insertion_50demos/${STEP}/params/"
                rsync -avz --progress \
                    "${HPC}:${LATEST}/assets/" \
                    "${LOCAL_DEST}/${config}/peg_insertion_50demos/${STEP}/assets/" 2>/dev/null || true
                echo "    ✓ ${config} step ${STEP} params downloaded"
            else
                echo "    ✗ ${config}: no checkpoint found"
            fi
        done
        ;;
    *)
        echo "  Downloading all (default)..."
        rsync -avz --progress "${HPC}:${HPC_CKPT}/" "${LOCAL_DEST}/"
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Downloaded to: ${LOCAL_DEST}/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Serve π0-FAST for inference (recommended - fastest):"
echo "     cd openpi_ur5e/openpi-ur5e"
echo "     uv run python scripts/serve_policy.py --port 8000 \\"
echo "       policy:checkpoint \\"
echo "       --policy.config=pi0_fast_ur5e_peg_insertion_lora \\"
echo "       --policy.dir=checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999"
echo ""
echo "  2. Serve π0:"
echo "     uv run python scripts/serve_policy.py --port 8000 \\"
echo "       policy:checkpoint \\"
echo "       --policy.config=pi0_ur5e_peg_insertion_lora \\"
echo "       --policy.dir=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4999"
echo ""
echo "  3. Serve π0.5:"
echo "     uv run python scripts/serve_policy.py --port 8000 \\"
echo "       policy:checkpoint \\"
echo "       --policy.config=pi05_ur5e_peg_insertion_lora \\"
echo "       --policy.dir=checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999"
echo ""
