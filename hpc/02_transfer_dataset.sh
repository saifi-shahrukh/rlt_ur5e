#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Transfer Dataset to HPC
# Run on: LOCAL machine (lab workstation)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Transfer Dataset → HPC"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Configuration ───────────────────────────────────────────────────────────
HPC="saifi@hpc-headnode.iis.fhg.de"
HPC_DEST="/data/beegfs/home/saifi/datasets/saifi/ur5e-peg-insertion-dual"

# Find local dataset
LOCAL_PATHS=(
    "${HOME}/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual"
    "/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual"
    "${HOME}/rlt_ur5e/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual"
)

LOCAL=""
for p in "${LOCAL_PATHS[@]}"; do
    if [[ -d "$p" ]]; then
        LOCAL="$p"
        break
    fi
done

if [[ -z "${LOCAL}" ]]; then
    echo "ERROR: Dataset not found locally!"
    echo "Searched:"
    for p in "${LOCAL_PATHS[@]}"; do echo "  - $p"; done
    echo ""
    echo "Set LOCAL= manually in this script or provide path:"
    echo "  bash 02_transfer_dataset.sh /path/to/ur5e-peg-insertion-dual"
    exit 1
fi

# Allow override via argument
[[ -n "${1}" ]] && LOCAL="${1}"

SIZE=$(du -sh "${LOCAL}" | cut -f1)
FILES=$(find "${LOCAL}" -type f | wc -l)

echo "  Source: ${LOCAL}"
echo "  Dest:   ${HPC}:${HPC_DEST}"
echo "  Size:   ${SIZE} (${FILES} files)"
echo ""
read -p "  Transfer? [y/N] " -n 1 -r
echo ""

[[ ! $REPLY =~ ^[Yy]$ ]] && echo "Aborted." && exit 0

echo ""
rsync -avz --progress "${LOCAL}/" "${HPC}:${HPC_DEST}/"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Dataset transferred!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Now on HPC: cd rlt_ur5e/hpc && bash 03_train.sh all"
echo ""
