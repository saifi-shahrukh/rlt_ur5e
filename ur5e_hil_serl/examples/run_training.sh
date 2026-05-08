#!/bin/bash
# =============================================================================
# HIL-SERL Training Launch Script
# =============================================================================
# Usage:
#   ./run_training.sh learner    # Terminal 1: Start the learner (GPU)
#   ./run_training.sh actor      # Terminal 2: Start the actor (CPU + robot)
#   ./run_training.sh kill       # Kill leftover processes and free ports
# =============================================================================

set -e
cd "$(dirname "$0")"

# Activate venv
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

# Configuration
EXP_NAME="peg_insertion"
DEMO_FILE="./demo_data/peg_insertion_608_transitions_2026-05-04_17-26-14.pkl"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "${1}" in
    learner)
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Starting LEARNER (GPU) on ports 5588/5589${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python train_rlpd.py \
            --exp_name "$EXP_NAME" \
            --demo_path "$DEMO_FILE" \
            --learner
        ;;

    actor)
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Starting ACTOR (CPU + Robot)${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${YELLOW}Note: Make sure the LEARNER is running first!${NC}"
        echo ""
        python train_rlpd.py \
            --exp_name "$EXP_NAME" \
            --actor
        ;;

    kill)
        echo -e "${YELLOW}Killing leftover processes...${NC}"
        # Kill train_rlpd processes
        pkill -f "train_rlpd.py" 2>/dev/null && echo "  Killed train_rlpd.py" || echo "  No train_rlpd.py running"
        # Free ports
        for port in 5588 5589; do
            pids=$(lsof -ti:$port 2>/dev/null || true)
            if [ -n "$pids" ]; then
                echo -e "  ${RED}Killing PIDs on port $port: $pids${NC}"
                echo "$pids" | xargs kill -9 2>/dev/null || true
            else
                echo -e "  ${GREEN}Port $port is free${NC}"
            fi
        done
        echo -e "${GREEN}Done!${NC}"
        ;;

    *)
        echo "Usage: $0 {learner|actor|kill}"
        echo ""
        echo "  learner  - Start the learner (GPU, Terminal 1) — start this FIRST"
        echo "  actor    - Start the actor (CPU + robot, Terminal 2)"
        echo "  kill     - Kill leftover processes and free ports 5588/5589"
        echo ""
        echo "Typical workflow:"
        echo "  Terminal 1:  ./run_training.sh learner"
        echo "  Terminal 2:  ./run_training.sh actor"
        echo ""
        echo "If you get port errors:"
        echo "  ./run_training.sh kill"
        exit 1
        ;;
esac
