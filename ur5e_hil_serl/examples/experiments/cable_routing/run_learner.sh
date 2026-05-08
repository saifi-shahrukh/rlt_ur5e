#!/bin/bash
# Run the learner for cable routing task
# Usage: cd ur5e_hil_serl/examples && bash experiments/cable_routing/run_learner.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/serl_robot_infra:${SCRIPT_DIR}/../..:$PYTHONPATH"

cd "$SCRIPT_DIR/../.."

python train_rlpd.py \
    --exp_name cable_routing \
    --learner \
    --demo_path "${SCRIPT_DIR}/demos/cable_routing_demos.pkl" \
    --checkpoint_path "${SCRIPT_DIR}/checkpoints"
