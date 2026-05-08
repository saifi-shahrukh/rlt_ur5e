#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run VLA-Only Inference (baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════
# This runs the VLA directly on the robot using the lerobot inference script.
# Use this to measure the VLA-only baseline success rate.
#
# Prerequisites: VLA server running (scripts/start_vla_server.sh)
# ═══════════════════════════════════════════════════════════════════════════
set -e

LEROBOT_DIR="/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello"

PROMPT="Pick up the peg and insert it into the hole."
ROBOT_IP="172.22.1.139"
VLA_IP="localhost"
VLA_PORT=8000
FPS=30

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  VLA-Only Baseline Inference"
echo "  Policy: π0-FAST (4-demo fine-tuned)"
echo "  Prompt: ${PROMPT}"
echo "  Robot:  ${ROBOT_IP}"
echo "  Server: ws://${VLA_IP}:${VLA_PORT}"
echo "═══════════════════════════════════════════════════════════════"
echo "  Press ESC to stop. Keep hand on E-STOP!"
echo ""

cd "${LEROBOT_DIR}"
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

exec python scripts/remote_pi_inference_dual_cam.py \
    --ip=${VLA_IP} \
    --port=${VLA_PORT} \
    --prompt="${PROMPT}" \
    --robot.type=ur5e_dual_cam \
    --robot.ip=${ROBOT_IP} \
    --fps=${FPS}
