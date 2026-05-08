#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run RLT Training (VLA + RL Token + SAC residual learning)
# ═══════════════════════════════════════════════════════════════════════════
# Prerequisites: VLA server running (scripts/start_vla_server.sh)
# ═══════════════════════════════════════════════════════════════════════════
set -e

PROJECT_ROOT="/home/robolab-2/ur5e_hande_workspace/rlt_ur5e"
cd "${PROJECT_ROOT}"

# Activate venv
source ur5e_hil_serl/.venv/bin/activate

# Set paths
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ur5e_hil_serl:${PROJECT_ROOT}/ur5e_hil_serl/serl_robot_infra:${PROJECT_ROOT}/ur5e_hil_serl/examples:${PYTHONPATH}"

# SAC uses tiny MLPs — CPU is fine (avoids JAX/PyTorch GPU conflict)
export JAX_PLATFORMS=cpu

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RLT Training — Peg Insertion"
echo "  VLA: π0-FAST (4-demo fine-tuned)"
echo "  RL Token: checkpoints/rl_token/peg_insertion_9demos_v1.pt"
echo "  SAC: 2-layer MLP [256,256], ensemble=2 (TD3)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Pass through all args (e.g. --fake_env, --warmup_only, etc.)
exec python -m rlt.examples.peg_insertion.train_rlt "$@"
