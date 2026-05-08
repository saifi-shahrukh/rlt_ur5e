# 08 — UR Controller Setup

## Overview

The `UR5eImpedanceController` (`robot_controllers/ur5e_controller.py`) is an asynchronous impedance controller that:
- Runs a **100Hz control loop** in a separate thread
- Uses UR's `forceMode()` for compliant motion
- Manages the Robotiq Hand-E gripper via async Modbus TCP
- Handles resets (retract → moveJ → restart forceMode)

## Architecture

```
UR5eImpedanceController (Thread)
  ├── RTDEControlInterface (ur-rtde) — sends commands at 100Hz
  ├── RTDEReceiveInterface (ur-rtde) — reads robot state
  ├── HandEGripper (async Modbus TCP) — gripper control
  └── Control Loop:
        1. Read robot state (TCP pose, forces, joint angles)
        2. Compute spring-damper force from position error
        3. Send via forceMode() at 100Hz
        4. Check for truncation (excessive force)
        5. Handle reset requests
```

## Impedance Control via forceMode

The controller converts position commands into forces using a spring-damper model:

```python
force = Kp * position_error + Kd * velocity_error
```

Where:
- **Kp** (stiffness) ≈ calculated from `ERROR_DELTA` config
- **Kd** (damping) ≈ calculated from `FORCEMODE_DAMPING` config
- `forceMode()` applies these forces while maintaining compliance

## Key Configuration

```python
# In experiments/peg_insertion/config.py:
ERROR_DELTA = 0.03              # Max position error for force calculation (m)
FORCEMODE_DAMPING = 0.1         # Damping (0=fast/stiff, 1=slow/soft)
CONTROLLER_HZ = 100             # Control loop frequency
FORCEMODE_LIMITS = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]  # Speed limits in forceMode
```

## Reset Sequence

When an episode ends (success or timeout):
1. `forceModeStop()` — exit force control
2. Retract peg upward 8cm via impedance (gentle)
3. `moveJ(HOME_Q)` — move to safe position
4. `moveJ(RESET_Q)` — move to episode start
5. `zeroFtSensor()` — zero the force/torque sensor
6. `forceModeSetDamping()` — restart force control
7. Resume 100Hz control loop

## UR Robot Requirements

### On the Teach Pendant:
1. **Disable EtherNet/IP adapter** (Settings → System → EtherNet/IP)
2. **Disable PROFINET** if active
3. **Unlock the robot** and activate remote control
4. Ensure no protective stops are active

### Network:
- Robot IP: `172.22.1.139`
- PC IP: `172.22.1.100` (same subnet)
- Firewall must allow RTDE ports (30003, 30004)

## Singularity Prevention

The UR5e's `forceMode()` has a **larger singularity avoidance envelope** than position mode. It refuses to execute if:
- TCP is within ~30cm of the base center (XY plane)
- J5 (Wrist 2) is near 0° or 180°
- Arm is fully extended

**Fix**: Use a tight safety box that keeps the TCP away from singularity zones. See `README_2.md`.

## Gripper Control

```python
# Gripper commands (async, non-blocking):
gripper.close(speed=255, force=150)   # Close
gripper.open(speed=255, force=100)    # Open
# Neutral (target_grip=0): no command sent, gripper holds position
```

For peg insertion: `GRIPPER_RELEASE_ON_RESET = False` keeps the gripper closed.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `RTDE input registers already in use` | Another RTDE connection active | Kill other processes, disable EtherNet/IP adapter |
| `FORCE MODE NOT POSSIBLE IN SINGULARITY` | TCP near base or wrist alignment | Tighten safety box |
| `RTDE control script is not running` | Protective stop or singularity | Clear on teach pendant, restart |
| `forceMode failed, recovering...` | UR rejected force command | Normal during exploration, reduces with tight box |
| Robot moves violently after reset | MRP clipping bug (now fixed) | Verify MRP-based clip_safety_box is active |
| Gripper doesn't respond | Modbus connection lost | Restart the actor |
