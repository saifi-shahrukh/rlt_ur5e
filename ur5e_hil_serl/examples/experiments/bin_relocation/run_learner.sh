#!/bin/bash
# Run the learner for bin relocation task
# Usage: cd ur5e_hil_serl/examples && bash experiments/bin_relocation/run_learner.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/serl_robot_infra:${SCRIPT_DIR}/../..:$PYTHONPATH"

cd "$SCRIPT_DIR/../.."

python train_rlpd.py \
    --exp_name bin_relocation \
    --learner \
    --demo_path "${SCRIPT_DIR}/demos/bin_relocation_demos.pkl" \
    --checkpoint_path "${SCRIPT_DIR}/checkpoints"
