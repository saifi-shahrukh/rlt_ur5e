#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Collect 50 Peg Insertion Demonstrations for RLT Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════
#
# This script records teleoperated demonstrations using the keyboard teleop.
# The dataset will be compatible with π0, π0-FAST, and π0.5 fine-tuning.
#
# Key Controls:
#   SPACE     → Start recording (move robot first, then press)
#   → (Right) → End & SAVE episode
#   ← (Left)  → DISCARD episode (re-record)
#   ESC       → Stop all recording
#   G         → Toggle gripper
#   W/S/A/D   → Move XY
#   Q/E       → Move Z (up/down)
#   I/K/J/L   → Rotate
#
# Usage:
#   cd ~/ur5e_hande_workspace/rlt_ur5e
#   bash scripts/collect_50_demos.sh
#
#   # To resume (add more episodes to existing dataset):
#   bash scripts/collect_50_demos.sh --resume
#
# ═══════════════════════════════════════════════════════════════════════════
set -e

# Configuration
DATASET_REPO="saifi/ur5e-peg-insertion-dual"
TASK="Pick up the peg and insert it into the hole."
NUM_EPISODES=50
FPS=30
EPISODE_TIME_S=60
RESET_TIME_S=30
ROBOT_IP="172.22.1.139"

# Dataset storage location
DATASET_ROOT="/home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets"

# Move to lerobot workspace
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello

# Activate the lerobot venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found at $(pwd)/.venv"
    echo "Run: cd $(pwd) && uv sync"
    exit 1
fi

# Ensure libfreenect2 is available for Kinect v2
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

# Suppress terminal echo issues with keyboard teleop
stty -echo 2>/dev/null
trap 'stty echo 2>/dev/null' EXIT INT TERM

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  DEMO COLLECTION: Peg Insertion (50 episodes)"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Dataset: ${DATASET_REPO}"
echo "  Task:    ${TASK}"
echo "  FPS:     ${FPS}"
echo "  Robot:   ${ROBOT_IP}"
echo "  Storage: ${DATASET_ROOT}/${DATASET_REPO}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Build the command
CMD="python scripts/record.py \
    --robot.type=ur5e_dual_cam \
    --robot.ip=${ROBOT_IP} \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=${ROBOT_IP} \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=${DATASET_REPO} \
    --dataset.single_task=\"${TASK}\" \
    --dataset.root=${DATASET_ROOT} \
    --dataset.num_episodes=${NUM_EPISODES} \
    --dataset.fps=${FPS} \
    --dataset.episode_time_s=${EPISODE_TIME_S} \
    --dataset.reset_time_s=${RESET_TIME_S} \
    --dataset.push_to_hub=False \
    --dataset.video=True"

# Add --resume if requested
if [[ "$*" == *"--resume"* ]]; then
    CMD="${CMD} --resume"
    echo "  ⟳ RESUMING existing dataset"
    echo ""
fi

# Execute
echo "  Running: python scripts/record.py ..."
echo ""

# Use eval to handle the quoted task string
eval $CMD
