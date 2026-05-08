# LeRobot-UR5e: Demo Collection & VLA-Only Inference

## Purpose
Collects teleoperated demonstrations (dual camera) and runs VLA-only inference on the robot using the policy server.

## Architecture
```
lerobot_ur5e_gello/
├── scripts/
│   ├── record.py                        ← Record demos (keyboard teleop)
│   ├── record_wrapper.sh                ← Wrapper with libfreenect2 setup
│   └── remote_pi_inference_dual_cam.py  ← VLA-only inference on robot
├── lerobot_robot_ur5e/                  ← UR5e robot plugin
│   └── lerobot_robot_ur5e/
│       ├── config_ur5e.py               ← UR5EConfig, UR5EDualCamConfig
│       ├── ur5e.py                      ← Robot control (RTDE + gripper)
│       └── robotiq_gripper.py           ← Robotiq Hand-E driver
├── lerobot_camera_kinect/               ← Kinect v2 Xbox camera plugin
├── lerobot_teleoperator_keyboard_ur5e/  ← Keyboard teleop (cartesian)
├── lerobot_teleoperator_gello/          ← GELLO teleop (not used currently)
└── openpi_client/                       ← WebSocket client for VLA server
```

## Demo Collection

```bash
cd openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

python scripts/record.py \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.root=/path/to/datasets \
    --dataset.num_episodes=50 \
    --dataset.fps=30 \
    --dataset.push_to_hub=False
```

### Recording Controls
| Key | Action |
|-----|--------|
| SPACE | Start recording |
| → (Right) | End & save episode |
| ← (Left) | Discard episode |
| ESC | Stop all recording |
| G | Toggle gripper |
| W/S/A/D | Move XY |
| Q/E | Move Z |
| I/K/J/L/U/O | Rotate |

## VLA-Only Inference

```bash
python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 --fps=30
```

## Config Options

| Config Field | Default | Description |
|-------------|---------|-------------|
| `calibrate_gripper` | `False` | Skip gripper calibration (peg already grasped) |
| `freedrive` | `False` | Enable freedrive mode |
| `ip` | `172.22.1.139` | Robot IP address |

## Cameras
| Camera | Type | Serial | Mount |
|--------|------|--------|-------|
| wrist_cam | RealSense D435 | 034422070605 | Wrist |
| overview_cam | Kinect v2 Xbox | 000631452147 | Overhead |

## Venv
- **Path:** `openpi_ur5e/lerobot_ur5e_gello/.venv/`
- **Python:** 3.11
- **Key packages:** lerobot, ur_rtde, pyrealsense2, freenect2
