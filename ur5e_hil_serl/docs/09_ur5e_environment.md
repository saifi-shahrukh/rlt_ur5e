# 09 — UR5e Environment (`ur5e_env`) Setup

## Overview

The UR5e environment (`serl_robot_infra/ur_env/`) is a Gymnasium-compatible environment that provides the interface between the RL algorithm and the physical robot.

## Directory Structure

```
serl_robot_infra/ur_env/
├── envs/
│   ├── ur5e_env.py        # Base UR5e Gymnasium environment
│   ├── wrappers.py        # Intervention, reward classifier, gripper wrappers
│   └── relative_env.py    # RelativeFrame wrapper (actions in TCP frame)
├── camera/
│   ├── rs_capture.py      # Intel RealSense D435 capture
│   ├── kinect_capture.py  # Kinect Xbox v2 capture
│   └── video_capture.py   # Generic video capture
├── spacemouse/
│   ├── fake_spacemouse.py     # Keyboard-based teleoperation
│   ├── spacemouse_expert.py   # Real SpaceMouse input
│   └── keyboard_expert.py     # Keyboard expert actions
└── utils/
    ├── hande_gripper.py   # Robotiq Hand-E async driver
    └── rotations.py       # Quaternion/MRP/euler utilities
```

## UR5eEnv (Base Environment)

**File**: `ur_env/envs/ur5e_env.py`

### Observation Space
```python
{
    'tcp_pose': Box(7,),        # [x, y, z, qx, qy, qz, qw]
    'tcp_vel': Box(6,),         # [vx, vy, vz, wx, wy, wz]
    'tcp_force': Box(3,),       # [fx, fy, fz]
    'tcp_torque': Box(3,),      # [tx, ty, tz]
    'gripper_pose': Box(1,),    # [0=open, 1=closed]
    'wrist_1': Box(H, W, 3),   # RealSense image
    'overview': Box(H, W, 3),  # Kinect image
}
```

### Action Space
```python
Box(7,)  # [dx, dy, dz, drx, dry, drz, grip]
# Position: meters per step (scaled by ACTION_SCALE[0])
# Rotation: MRP per step (scaled by ACTION_SCALE[1])
# Gripper: [-1, 0, 1] → open/hold/close
```

### Key Methods
- `reset()`: Triggers controller reset sequence, returns initial observation
- `step(action)`: Applies action, returns (obs, reward, done, truncated, info)
- `clip_safety_box(pose)`: Clips target pose to safety limits (MRP-based)

## Wrapper Stack

The full wrapper stack (applied in order):

```python
env = PegInsertionEnv(config)           # Task-specific env
env = GripperCloseEnv(env)              # Forces gripper closed
env = KeyboardIntervention(env)         # Human-in-the-loop
env = RelativeFrame(env)                # Actions in TCP frame
env = Quat2EulerWrapper(env)            # Quaternion → euler for proprio
env = SERLObsWrapper(env, proprio_keys) # Selects observation keys
env = ChunkingWrapper(env)              # Action chunking (horizon=1)
env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)  # Reward
env = RecordEpisodeStatistics(env)      # Gymnasium stats
```

## Configuration

Each task has a config class inheriting from `DefaultEnvConfig`:

```python
class EnvConfig(DefaultEnvConfig):
    ROBOT_IP = "172.22.1.139"
    CONTROLLER_HZ = 100
    RESET_Q = np.deg2rad([33.56, -76.79, -132.20, -60.98, 90.22, 35.98])
    TARGET_POSE = np.array([0.362, 0.080, 0.085, 2.176, -2.266, 0.0])
    HOME_Q = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])
    ABS_POSE_LIMIT_LOW = np.array([0.28, -0.02, 0.03, -0.10, -0.10, -0.10])
    ABS_POSE_LIMIT_HIGH = np.array([0.42, 0.14, 0.20, 0.10, 0.10, 0.10])
    ACTION_SCALE = np.array([0.005, 0.03, 1.0])
    MAX_EPISODE_LENGTH = 100
```

## Safety Clipping (MRP-based)

Orientation is clipped as **Modified Rodrigues Parameters relative to reset pose**:

```python
def clip_safety_box(self, pose):
    # Position: absolute clipping
    pose[:3] = np.clip(pose[:3], xyz_low, xyz_high)
    
    # Orientation: relative MRP clipping
    orientation_diff = (R_current * R_reset.inv()).as_mrp()
    orientation_diff = np.clip(orientation_diff, mrp_low, mrp_high)
    pose[3:] = (R.from_mrp(orientation_diff) * R_reset).as_quat()
```

This prevents the euler-angle bug that causes violent robot movements.

## Camera Setup

### RealSense D435 (Wrist)
```python
REALSENSE_CAMERAS = {
    "wrist_1": {
        "serial_number": "034422070605",
        "dim": (640, 480),
        "exposure": 40000,
    },
}
```

### Kinect v2 (Overview)
```python
KINECT_CAMERAS = {
    "overview": "000631452147",
}
```

## Testing the Environment

```bash
# Unit tests (no robot needed)
python -m pytest tests/test_configs.py tests/test_rotations.py -v

# With robot connected:
python -m pytest tests/test_env.py -v
```
