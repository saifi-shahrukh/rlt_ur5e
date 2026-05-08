# UR5e HIL-SERL Quick Start Guide

**Human-in-the-Loop Sample-Efficient RL for UR5e + Hand-E Gripper**

This project enables learning robotic manipulation tasks (peg insertion, PCB
assembly, cable routing, bin relocation) on a UR5e with a Robotiq Hand-E
gripper using human-in-the-loop reinforcement learning.

---

## Hardware Setup

| Component | Model | Connection |
|-----------|-------|------------|
| Robot | UR5e | Ethernet (172.22.1.139) |
| Gripper | Robotiq Hand-E | UR Tool port |
| Wrist Camera | Intel RealSense D435 | USB 3.0 (serial: 034422070605) |
| Overview Camera | Kinect Xbox v2 | USB 3.0 (serial: 000631452147) |
| GPU | NVIDIA (CUDA 12) | For JAX/training |
| Input | Keyboard (FakeSpaceMouse) | For teleoperation |

---

## Installation

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl

# Create virtual environment (Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate

# Install all packages
pip install -e .
pip install -e serl_launcher

# Verify
python -c "import jax; print(jax.devices())"  # Should show CudaDevice
python -m pytest tests/ -q  # Should show 68 passed
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  record_demos.py / run_actor.py / run_learner.py        │
├─────────────────────────────────────────────────────────┤
│  Wrappers: ChunkingWrapper → SERLObsWrapper →           │
│            Quat2EulerWrapper → RelativeFrame →          │
│            KeyboardIntervention → GripperCloseEnv        │
├─────────────────────────────────────────────────────────┤
│  PegInsertionEnv (examples/experiments/peg_insertion/)   │
│    └── UR5eEnv (serl_robot_infra/ur_env/envs/ur5e_env) │
├─────────────────────────────────────────────────────────┤
│  UrImpedanceController (robot_controllers/ur5e_ctrl)    │
│    ├── ur-rtde forceMode (100Hz impedance loop)         │
│    ├── HandEGripper (async Modbus TCP)                  │
│    └── Reset: forceModeStop→speedL↑→moveJ→forceMode    │
├─────────────────────────────────────────────────────────┤
│  UR5e Robot (172.22.1.139)                              │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **MRP-based safety clipping**: Orientation is clipped as Modified Rodrigues
   Parameters (MRP) *relative to the reset pose* — NOT absolute euler angles.
   This prevents destroying the tool orientation when the robot is far from
   the world origin.

2. **Controller-owned reset**: The reset sequence (retract → moveJ → restart
   forceMode) runs entirely inside the controller thread. The env just sets
   `_reset` and waits.

3. **Impedance via forceMode**: Position commands are converted to
   spring-damper forces (Kp=10000, Kd=2200) and sent via ur-rtde `forceMode()`
   at 100Hz. This gives compliant behavior during teleoperation.

---

## Task: Peg Insertion

### Step 1: Record Demonstrations

The peg must be pre-grasped in the gripper. The gripper stays closed.

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python record_demos.py --exp_name peg_insertion --successes_needed 20
```

**Controls:**
- Arrow keys: XY movement
- `1` / `0`: Z up / down
- Right Ctrl: toggle gripper (not needed for peg)
- Episode ends on timeout (100 steps) or when peg reaches target

Demos are saved to `./demo_data/peg_insertion_20_demos_<timestamp>.pkl`

### Step 2: Train Reward Classifier (Optional)

If you want learned success detection instead of distance-based:

```bash
python train_classifier.py --exp_name peg_insertion \
    --demo_path ./demo_data/peg_insertion_20_demos_*.pkl
```

Saves to `./classifier_ckpt/`

### Step 3: Train RL Policy

Terminal 1 (Learner — runs on GPU):
```bash
python run_learner.py --exp_name peg_insertion \
    --demo_path ./demo_data/peg_insertion_20_demos_*.pkl
```

Terminal 2 (Actor — runs on robot):
```bash
python run_actor.py --exp_name peg_insertion
```

The human can intervene via keyboard at any time during training.

---

## Configuration Reference

All task configs are in `examples/experiments/<task>/config.py`.

### Key Parameters

| Parameter | Description | Peg Value |
|-----------|-------------|----------|
| `ROBOT_IP` | UR5e IP address | `172.22.1.139` |
| `RESET_Q` | Joint angles for episode start (radians) | `deg2rad([33.5, -74.6, -129.6, -65.7, 90.2, 34.7])` |
| `TARGET_POSE` | TCP pose when task is complete [x,y,z,rx,ry,rz] | `[0.3615, -0.0794, 0.086, 2.2, -2.243, 0.0]` |
| `ABS_POSE_LIMIT_LOW/HIGH` | Safety box [x,y,z, mrp_x,mrp_y,mrp_z] | See below |
| `ACTION_SCALE` | [pos_m/step, rot_rad/step, grip_scale] | `[0.01, 0.05, 1.0]` |
| `ERROR_DELTA` | Max position error for force calc (m) | `0.03` |
| `FORCEMODE_DAMPING` | Force mode damping (0=fast, 1=slow) | `0.1` |
| `GRIPPER_RELEASE_ON_RESET` | Open gripper during reset? | `False` (peg) |

### Safety Box Explained

```python
ABS_POSE_LIMIT_LOW  = np.array([X_min, Y_min, Z_min, MRP_x_min, MRP_y_min, MRP_z_min])
ABS_POSE_LIMIT_HIGH = np.array([X_max, Y_max, Z_max, MRP_x_max, MRP_y_max, MRP_z_max])
```

- **Position (first 3)**: Absolute TCP position limits in robot base frame (meters)
- **Orientation (last 3)**: Maximum rotation deviation from reset pose as MRP
  - ±0.1 MRP ≈ ±22° rotation
  - ±0.3 MRP ≈ ±66° rotation
  - For peg insertion: use ±0.1 (small deviations only)

### How to Measure RESET_Q and TARGET_POSE

1. **RESET_Q**: Use the UR teach pendant to jog the robot to the desired start
   position (peg ~5cm above hole). Read joint angles from the pendant.
   Convert degrees to radians: `np.deg2rad([j1, j2, j3, j4, j5, j6])`

2. **TARGET_POSE**: Use the teach pendant to jog the peg into the hole.
   Read TCP pose (Position tab). The format is `[x, y, z, rx, ry, rz]` where
   rx/ry/rz are axis-angle (rotation vector) components.

---

## Troubleshooting

### "RTDE control script is not running!"

This means the ur-rtde control script died. Common causes:
- Robot hit a singularity (J5 near 0° or 180°)
- Protective stop triggered (excessive force/speed)
- Safety box too tight (orientation clipping sends bad target)

**Fix**: Clear the protective stop on the teach pendant, then restart.

### Robot makes violent movement after reset

This was the original bug — caused by `clip_safety_box` using absolute euler
angle clipping instead of MRP-relative clipping. Now fixed.

### "forcemode failed, is now truncated!"

The `_truncate_check()` detected downward force > 20N. This usually means:
- The peg is jammed / the robot is pushing too hard
- The episode will auto-reset

### Gripper doesn't stay closed

Ensure `GRIPPER_RELEASE_ON_RESET = False` in config and the wrapper stack
includes `GripperCloseEnv`.

### Keyboard not responding

The FakeSpaceMouse uses `pynput` which requires the terminal to have focus.
Click on the terminal window before pressing keys.

---

## File Structure

```
ur5e_hil_serl/
├── examples/
│   ├── record_demos.py          # Record human demonstrations
│   ├── run_actor.py             # RL actor (robot interaction)
│   ├── run_learner.py           # RL learner (GPU training)
│   ├── train_classifier.py     # Train reward classifier
│   ├── experiments/
│   │   ├── config.py            # Base training config
│   │   ├── mappings.py          # Task name → config mapping
│   │   ├── peg_insertion/       # Peg insertion task
│   │   │   ├── config.py        # Hardware + RL config
│   │   │   └── wrapper.py       # Task-specific env wrapper
│   │   ├── pcb_insertion/       # PCB insertion task
│   │   ├── cable_routing/       # Cable routing task
│   │   └── bin_relocation/      # Bin relocation task
│   └── demo_data/               # Saved demonstrations
├── serl_robot_infra/
│   ├── robot_controllers/
│   │   └── ur5e_controller.py   # Impedance controller (100Hz)
│   └── ur_env/
│       ├── envs/
│       │   ├── ur5e_env.py      # Base Gymnasium environment
│       │   ├── wrappers.py      # SpaceMouse, Quat2Euler, etc.
│       │   └── relative_env.py  # RelativeFrame wrapper
│       ├── camera/              # RealSense + Kinect capture
│       ├── spacemouse/          # SpaceMouse + FakeSpaceMouse
│       └── utils/               # Gripper, rotations, etc.
├── serl_launcher/               # RL algorithms (SAC, BC, DrQ)
├── tests/                       # Unit + integration tests
└── QUICKSTART.md                # This file
```

---

## Development Workflow

```bash
# Run all tests
python -m pytest tests/ -q

# Run only unit tests (no GPU needed)
python -m pytest tests/ --ignore=tests/test_pipeline.py -q

# Run integration tests (needs GPU)
python -m pytest tests/test_pipeline.py -q
```

---

## Key Bug Fixes Applied

1. **MRP safety clipping** (`ur5e_env.py`): Changed `clip_safety_box()` from
   absolute euler angle clipping to MRP-relative-to-reset-pose clipping.
   Without this, the robot's orientation was destroyed on the first step.

2. **`isProgramRunning()` recovery** (`ur5e_controller.py`): After `moveJ()`,
   check if the RTDE script is still running. If not, reconnect.

3. **Always clear `_reset`** (`ur5e_controller.py`): Prevents `move_to_joints()`
   from blocking forever if the reset sequence fails.

4. **Step orientation as MRP** (`ur5e_env.py`): Action rotation uses
   `R.from_mrp(action * scale / 4)` matching the ur5e_serl convention.
