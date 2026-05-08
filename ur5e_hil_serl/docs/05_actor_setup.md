# 05 — Actor Setup

## Overview

The **actor** is the process that controls the physical robot. It:
1. Connects to the UR5e robot via RTDE
2. Initializes cameras (RealSense + Kinect)
3. Runs the RL policy to select actions
4. Steps the environment (moves the robot)
5. Sends transitions to the learner via ZMQ
6. Receives updated network weights from the learner

The actor runs on **CPU only** (GPU is reserved for the learner).

## Prerequisites

- Learner is already running (see [06_training_pipeline.md](06_training_pipeline.md))
- Robot is powered on, no protective stops
- Cameras are connected
- No other RTDE connections active on the robot

## Running the Actor

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python train_rlpd.py --exp_name peg_insertion --actor
```

### What happens automatically:
- `CUDA_VISIBLE_DEVICES=""` is set (forces CPU)
- `JAX_PLATFORMS="cpu"` is set
- CUDA warnings are suppressed

## Actor Architecture

```
train_rlpd.py --actor
  ├── Creates env (PegInsertionEnv + wrappers)
  │     ├── UR5eImpedanceController (100Hz thread)
  │     ├── RealSense D435 (wrist camera)
  │     ├── Kinect v2 (overview camera)
  │     └── FakeSpaceMouse (keyboard intervention)
  ├── Creates SAC agent (on CPU)
  ├── Connects to learner via TrainerClient (ZMQ)
  └── Runs actor loop:
        1. Sample action from policy
        2. env.step(action) → move robot
        3. Store transition
        4. Send to learner
        5. Receive updated weights
```

## Human Intervention During Training

The actor supports **human-in-the-loop** corrections:

- Use the **arrow keys** to override the policy's action
- When you press a key, the robot follows YOUR command instead of the policy
- These interventions are stored in a separate buffer and used for training
- **Intervene early and often** in the first 10-20 minutes
- Gradually reduce interventions as the policy improves

## Configuration

Key actor parameters in `experiments/peg_insertion/config.py`:

```python
max_steps = 1_000_000       # Total actor steps
random_steps = 0            # Steps of random actions before policy (0 = use demos)
buffer_period = 1000        # Save buffer to disk every N steps
log_period = 100            # Log stats every N steps
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA_ERROR_NO_DEVICE` warnings | Harmless — JAX falls back to CPU correctly |
| `IndexError: list index out of range` | No buffer files exist yet — fixed in latest code |
| `RTDE input registers already in use` | Another process has RTDE connection. Kill it: `pkill -f train_rlpd` |
| Robot doesn't move | Check teach pendant for protective stops |
| `forceMode failed` | Safety box too wide — see README_2.md |
| Keyboard not responding | Click on the terminal window (pynput needs focus) |
