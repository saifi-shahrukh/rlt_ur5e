#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Download Checkpoints from HPC
# Run on: LOCAL machine
# ═══════════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Download Checkpoints ← HPC"
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
ssh "${HPC}" "find ${HPC_CKPT} -maxdepth 3 -type d 2>/dev/null | sort" || echo "  (connection failed)"
echo ""

echo "  Download options:"
echo "    1) All (π0 + π0.5)"
echo "    2) π0 only"
echo "    3) π0.5 only"
read -p "  Choice [1-3]: " CHOICE
echo ""

case "${CHOICE}" in
    1) SRC="${HPC_CKPT}/" ;;
    2) SRC="${HPC_CKPT}/pi0_ur5e_peg_insertion_lora/" ;;
    3) SRC="${HPC_CKPT}/pi05_ur5e_peg_insertion_lora/" ;;
    *) SRC="${HPC_CKPT}/" ;;
esac

echo "  From: ${HPC}:${SRC}"
echo "  To:   ${LOCAL_DEST}/"
echo ""

rsync -avz --progress "${HPC}:${SRC}" "${LOCAL_DEST}/"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Downloaded to: ${LOCAL_DEST}/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  To serve π0:"
echo "    .venv/bin/python scripts/serve_policy.py --port 8000 \\"
echo "      policy:checkpoint --policy.config=pi0_ur5e_peg_insertion_lora \\"
echo "      --policy.dir=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_9demos/29999"
echo ""
echo "  To serve π0.5:"
echo "    .venv/bin/python scripts/serve_policy.py --port 8000 \\"
echo "      policy:checkpoint --policy.config=pi05_ur5e_peg_insertion_lora \\"
echo "      --policy.dir=checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_9demos/29999"
echo ""
