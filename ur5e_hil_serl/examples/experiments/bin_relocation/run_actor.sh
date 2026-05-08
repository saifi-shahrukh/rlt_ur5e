#!/bin/bash
# Run the actor for bin relocation task
# Usage: cd ur5e_hil_serl/examples && bash experiments/bin_relocation/run_actor.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/serl_robot_infra:${SCRIPT_DIR}/../..:$PYTHONPATH"

cd "$SCRIPT_DIR/../.."

python train_rlpd.py \
    --exp_name bin_relocation \
    --actor \
    --ip localhost \
    --checkpoint_path "${SCRIPT_DIR}/checkpoints" \
    --save_video
