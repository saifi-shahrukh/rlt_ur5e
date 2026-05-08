#!/bin/bash
# Ensure libfreenect2 is available for Kinect v2 camera
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

stty -echo 2>/dev/null
trap 'stty echo 2>/dev/null' EXIT INT TERM
python scripts/record.py "$@"
