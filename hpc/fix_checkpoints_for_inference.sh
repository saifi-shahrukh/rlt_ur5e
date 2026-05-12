#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Fix checkpoints for inference/extraction
#
# Problem: Training used asset_id="saifi/ur5e-peg-insertion-dual" but the
# config now expects "saifi/ur5e-peg-insertion-50demos-v2".
# The checkpoint's assets/ dir has the old name (or is missing).
#
# Fix: Copy norm_stats.json into the checkpoint's assets directory
# under BOTH possible names so inference works regardless of config.
# ═══════════════════════════════════════════════════════════════════════════════
set -e

OPENPI="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e"
ASSETS_SRC="${OPENPI}/assets"

echo "═══════════════════════════════════════════════════════════════"
echo "  Fixing checkpoint assets for inference"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Find all checkpoints
for ckpt_dir in ${OPENPI}/checkpoints/*/peg_insertion_50demos/4999; do
    if [[ -d "${ckpt_dir}" ]]; then
        config=$(echo ${ckpt_dir} | sed 's|.*/checkpoints/||' | cut -d/ -f1)
        echo "  Processing: ${config} (step 4999)"
        
        # Create assets dir structure for both possible dataset names
        for dataset_name in "saifi/ur5e-peg-insertion-dual" "saifi/ur5e-peg-insertion-50demos-v2"; do
            target_dir="${ckpt_dir}/assets/${dataset_name}"
            mkdir -p "${target_dir}"
            
            # Find norm_stats source
            norm_src=""
            if [[ -f "${ASSETS_SRC}/${config}/${dataset_name}/norm_stats.json" ]]; then
                norm_src="${ASSETS_SRC}/${config}/${dataset_name}/norm_stats.json"
            elif [[ -f "${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-dual/norm_stats.json" ]]; then
                norm_src="${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-dual/norm_stats.json"
            elif [[ -f "${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json" ]]; then
                norm_src="${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json"
            fi
            
            if [[ -n "${norm_src}" && ! -f "${target_dir}/norm_stats.json" ]]; then
                cp "${norm_src}" "${target_dir}/norm_stats.json"
                echo "    ✓ Copied norm_stats → ${dataset_name}"
            elif [[ -f "${target_dir}/norm_stats.json" ]]; then
                echo "    ✓ Already exists: ${dataset_name}"
            else
                echo "    ✗ No norm_stats source found for ${dataset_name}"
            fi
        done
        echo ""
    fi
done

# Also fix step 4000 for pi0
for ckpt_dir in ${OPENPI}/checkpoints/*/peg_insertion_50demos/4000; do
    if [[ -d "${ckpt_dir}" ]]; then
        config=$(echo ${ckpt_dir} | sed 's|.*/checkpoints/||' | cut -d/ -f1)
        echo "  Processing: ${config} (step 4000)"
        
        for dataset_name in "saifi/ur5e-peg-insertion-dual" "saifi/ur5e-peg-insertion-50demos-v2"; do
            target_dir="${ckpt_dir}/assets/${dataset_name}"
            mkdir -p "${target_dir}"
            
            norm_src=""
            if [[ -f "${ASSETS_SRC}/${config}/${dataset_name}/norm_stats.json" ]]; then
                norm_src="${ASSETS_SRC}/${config}/${dataset_name}/norm_stats.json"
            elif [[ -f "${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-dual/norm_stats.json" ]]; then
                norm_src="${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-dual/norm_stats.json"
            elif [[ -f "${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json" ]]; then
                norm_src="${ASSETS_SRC}/${config}/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json"
            fi
            
            if [[ -n "${norm_src}" && ! -f "${target_dir}/norm_stats.json" ]]; then
                cp "${norm_src}" "${target_dir}/norm_stats.json"
                echo "    ✓ Copied norm_stats → ${dataset_name}"
            elif [[ -f "${target_dir}/norm_stats.json" ]]; then
                echo "    ✓ Already exists: ${dataset_name}"
            fi
        done
        echo ""
    fi
done

echo "  Done! Checkpoints now have norm_stats for inference."
