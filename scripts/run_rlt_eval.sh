#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Evaluate RLT Checkpoint (or VLA-only baseline)
# ═══════════════════════════════════════════════════════════════════════════
# Usage:
#   bash scripts/run_rlt_eval.sh                           # Eval best checkpoint
#   bash scripts/run_rlt_eval.sh path/to/ckpt.pkl 20       # Specific checkpoint, 20 eps
#   bash scripts/run_rlt_eval.sh --no_residual             # VLA-only (zero residual)
# ═══════════════════════════════════════════════════════════════════════════
set -e

PROJECT_ROOT="/home/robolab-2/ur5e_hande_workspace/rlt_ur5e"
cd "${PROJECT_ROOT}"

source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ur5e_hil_serl:${PROJECT_ROOT}/ur5e_hil_serl/serl_robot_infra:${PROJECT_ROOT}/ur5e_hil_serl/examples:${PYTHONPATH}"
export JAX_PLATFORMS=cpu

# Defaults
CHECKPOINT="checkpoints/rlt_runs/peg_insertion/best.pkl"
EPISODES=20
EXTRA_ARGS=""

# Parse args
if [[ "$1" == "--no_residual" ]]; then
    EXTRA_ARGS="--no_residual"
    echo "  Mode: VLA-only (zero residual)"
elif [[ -n "$1" && -f "$1" ]]; then
    CHECKPOINT="$1"
    [[ -n "$2" ]] && EPISODES="$2"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RLT Evaluation"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Episodes: ${EPISODES}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

exec python -m rlt.examples.peg_insertion.train_rlt \
  --eval_only \
  --eval_episodes ${EPISODES} \
  --checkpoint ${CHECKPOINT} \
  ${EXTRA_ARGS}
