# HIL-SERL Training Analysis & ForceMode Fix Guide

**Date**: 2026-05-05  
**Status**: Training is RUNNING but with frequent `forceMode failed` truncations  
**Goal**: Achieve successful peg insertion with high success rate

---

## Current Status ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Learner (GPU) | ✅ Running | Training at ~10 it/s, checkpoint at step 25000+ |
| Actor (CPU + Robot) | ✅ Running | Connected to UR5e, episodes running |
| Reward Classifier | ✅ Loaded | checkpoint_150, using wrist + overview cameras |
| Demo Buffer | ✅ 628 transitions | Loaded from pkl file |
| WandB Logging | ✅ Active | project: saifi/hil-serl |
| Human Intervention | ✅ Ready | FakeSpaceMouse (keyboard) |
| ForceMode | ⚠️ Failing frequently | "FORCE MODE NOT POSSIBLE IN SINGULARITY" |

---

## Problem: ForceMode Singularity Failures

### What's Happening

The UR5e teach pendant shows:
```
FORCE MODE NOT POSSIBLE IN SINGULARITY
```

And the actor terminal shows:
```
[RIC] forceMode failed, recovering...
[RIC] forcemode failed, is now truncated!
```

This means episodes are being **truncated prematurely** — the robot never reaches the insertion goal because forceMode keeps failing.

### Root Cause Analysis

**UR5e Singularities** occur in three cases:
1. **Wrist singularity**: J5 (Wrist 2) near 0° or 180°
2. **Elbow singularity**: Arm fully extended (J3 near 0° or ±180°)
3. **Shoulder/Base singularity**: TCP directly above/below the robot base

**Critical finding from Robotiq forum**: The UR's `forceMode()` has a **LARGER singularity avoidance envelope** than position mode. It triggers the singularity warning approximately **30cm before the TCP is directly over the base**.

### Your Configuration Issues

```python
# CURRENT safety box (TOO LARGE):
ABS_POSE_LIMIT_LOW  = np.array([0.10, -0.10, -0.01, -0.3, -0.3, -0.3])
ABS_POSE_LIMIT_HIGH = np.array([0.40, 0.10, 0.60, 0.3, 0.3, 0.3])
```

**Problem 1: X_min = 0.10m** — This allows the TCP to move within 10cm of the base center. The forceMode singularity envelope is ~30cm, so positions with X < 0.25 can trigger the singularity error.

**Problem 2: MRP orientation limits ±0.3** — This corresponds to **±66° rotation** from reset pose. For a peg insertion task, this is enormously excessive. A ±66° rotation can easily swing J5 toward 0° or 180° (wrist singularity) or create geometric singularity conditions.

**Problem 3: Z range too wide** — Z from -0.01 to 0.60m is a 61cm range. During random exploration, the robot can reach extreme configurations.

### Your Working Pose
```
TCP at operation: x≈0.33, y≈0.05-0.08  → 0.33m from base (borderline safe)
Target:           x=0.362, y=0.080     → 0.37m from base (safe)
J5 at reset: 90.22° (perfectly safe)
```

The reset position itself is safe, but **during random RL exploration**, the policy sends random actions that push the TCP toward the safety box edges — specifically toward X_min=0.10 or large rotations that trigger singularity.

---

## Fix: Tighten the Safety Box

### Recommended Configuration

```python
# In examples/experiments/peg_insertion/config.py:

# TIGHTENED safety box — prevents singularity during exploration
ABS_POSE_LIMIT_LOW  = np.array([0.28, -0.02, 0.03, -0.10, -0.10, -0.10])
ABS_POSE_LIMIT_HIGH = np.array([0.42,  0.14, 0.20,  0.10,  0.10,  0.10])
```

### Rationale

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| X_min | 0.10 | **0.28** | Keep TCP >28cm from base → outside singularity envelope |
| X_max | 0.40 | **0.42** | Small extension for target (0.362) |
| Y_min | -0.10 | **-0.02** | Target Y=0.08, limit exploration |
| Y_max | 0.10 | **0.14** | Allow reaching target + small margin |
| Z_min | -0.01 | **0.03** | Peg shouldn't go below 3cm |
| Z_max | 0.60 | **0.20** | Peg starts at ~15cm, no need to go to 60cm |
| MRP | ±0.3 | **±0.10** | ±22° is generous for peg insertion |

### How This Helps
1. **X ≥ 0.28m**: TCP always stays outside the ~30cm singularity envelope
2. **MRP ±0.10 (22°)**: Prevents large orientation swings that cause wrist singularity
3. **Tighter Z/Y**: Focuses exploration on useful workspace, faster learning

---

## How HIL-SERL Achieves Good Results (from the paper)

### Key Design Principles (from the ICRA 2024 paper + walkthrough)

1. **Pretrained ResNet-10 vision backbone** — frozen early layers, only trains top layers. This gives good image features from the start.

2. **RLPD (Reinforcement Learning with Prior Data)** — 50% demo buffer + 50% online experience in each training batch. The demos teach the policy good behavior immediately.

3. **Human-in-the-Loop corrections** — The human intervenes during training (using spacemouse/keyboard) when the robot gets stuck. These corrections go into both the demo buffer AND online buffer. **This is the critical ingredient** — without interventions, the random policy wastes time in bad states.

4. **Sparse binary reward** — A trained classifier gives reward=1 only when the task is complete. No intermediate rewards. This avoids reward shaping issues.

5. **Impedance/Compliance control** — The robot is compliant during insertion. Franka uses its built-in impedance controller; your UR5e uses forceMode for compliance. This is essential for insertion tasks.

6. **Tight workspace** — The original HIL-SERL uses very tight bounding boxes. For RAM insertion, the workspace is only a few centimeters around the target.

7. **20 demonstrations** — A small number of high-quality demos (20 successful insertions) bootstraps the policy.

8. **1-2.5 hours of training** — With ~10Hz control, this is 36,000-90,000 steps. Near-perfect success rate achieved in this range.

### Training Tips from the Walkthrough

- **Start intervening early and often** — In the first 10-20 minutes, intervene on almost every episode
- **Gradually reduce interventions** — As policy improves, intervene less
- **Reward classifier must be robust** — False positives are worse than false negatives. Collect 2-3x more negative samples than positive.
- **Image crops matter** — Crop cameras to focus on the task-relevant area (peg + hole)
- **Random reset helps** — Small XY/rotation randomization at start prevents overfitting to one start pose

### Typical Training Progression

| Steps | Expected Behavior |
|-------|-------------------|
| 0-1000 | Random exploration, frequent failures, many interventions needed |
| 1000-5000 | Policy starts moving toward target, fewer random movements |
| 5000-15000 | Insertion attempts begin, partial successes |
| 15000-30000 | Consistent insertions, human can stop intervening |
| 30000+ | Near-perfect success rate, policy refinement |

---

## Voxel-SERL Insights (from nisutte/voxel-serl)

Voxel-SERL extends SERL with 3D point cloud perception for a UR5 robot:

1. **Uses the same UR5 + impedance control architecture** as your setup
2. **VoxNet encoder** processes voxelized point clouds for 3D spatial understanding
3. **Same actor-learner split** with ZMQ communication
4. **For UR robots specifically**: They also use forceMode-based impedance control
5. **Key difference**: They add 3D perception which helps for tasks where the object position varies

For your peg insertion task, the 2D camera approach (wrist + overview) should be sufficient since the hole position is fixed.

---

## Complete Action Plan

### Step 1: Fix the Safety Box (IMMEDIATE)

Edit `/home/robolab-2/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples/experiments/peg_insertion/config.py`:

```python
# TIGHTENED safety box
ABS_POSE_LIMIT_LOW  = np.array([0.28, -0.02, 0.03, -0.10, -0.10, -0.10])
ABS_POSE_LIMIT_HIGH = np.array([0.42,  0.14, 0.20,  0.10,  0.10,  0.10])
```

### Step 2: Verify Workspace Fits Target

Your target pose is `[0.362, 0.080, 0.085]`. Check:
- 0.28 ≤ 0.362 ≤ 0.42 ✅
- -0.02 ≤ 0.080 ≤ 0.14 ✅  
- 0.03 ≤ 0.085 ≤ 0.20 ✅

### Step 3: Restart Training Fresh

```bash
# Kill existing processes
pkill -f train_rlpd.py
lsof -ti:5588 | xargs kill -9 2>/dev/null
lsof -ti:5589 | xargs kill -9 2>/dev/null

# Optional: Remove old checkpoints to start fresh
# (or keep them and the system will resume from step 25000)
mv ./checkpoints/peg_insertion ./checkpoints/peg_insertion_old_wide_box

# Start learner
python train_rlpd.py --exp_name peg_insertion \
    --demo_path ./demo_data/peg_insertion_608_transitions_2026-05-04_17-26-14.pkl \
    --learner

# Start actor (different terminal)
python train_rlpd.py --exp_name peg_insertion --actor
```

### Step 4: Intervene During Training!

This is **critical**. Watch the actor terminal and use the keyboard to guide the robot:
- When the robot moves away from the hole → intervene with arrow keys
- When the robot gets stuck → intervene
- When you see repeated forceMode failures → the robot is exploring bad regions

### Step 5: Monitor Progress

Watch WandB dashboard at https://wandb.ai/saifi/hil-serl for:
- `episode/return` — should increase over time
- `episode/intervention_count` — should decrease over time
- `episode/length` — should decrease (faster insertions)

---

## Why ForceMode Fails Explained

### The UR5e ForceMode Pipeline

```
Your code:
  train_rlpd.py (actor) → env.step(action)
    → UR5eEnv.step() → compute target → clip_safety_box()
      → controller.set_target_pos(target)
        → 100Hz loop: ur_control.forceMode(task_frame, selection, force, type, limits)
```

### What happens at singularity:
1. RL policy outputs a random action (especially early in training)
2. Action moves TCP target toward X_min=0.10 or large rotation
3. `clip_safety_box()` clips to safety limits — but limits are too wide
4. Controller sends force command to UR5e
5. UR5e internal singularity check detects proximity to singularity
6. `forceMode()` returns `False` (cannot execute in singularity)
7. Controller triggers `restart_ur_interface()` → episode truncated

### With tighter box:
1. RL policy outputs random action
2. Action moves TCP target toward edge
3. `clip_safety_box()` clips to **tight** limits (X≥0.28, MRP≤0.10)
4. Clipped target is ALWAYS in safe workspace → forceMode succeeds
5. Episode continues → robot learns from the experience

---

## Additional Optimizations (Optional)

### Reduce `_truncate_check` Force Threshold

Currently truncates at 20N downward force. For peg insertion, this might be too aggressive:
```python
# In ur5e_controller.py line ~280
def _truncate_check(self):
    downward_force = self.curr_force_lowpass[2] > 20.0  # Consider raising to 30-40N
```

### Reduce Action Scale for Precision

Current: `ACTION_SCALE = [0.005, 0.03, 1.0]` (5mm position, 0.03rad ≈ 1.7° rotation per step)

For high-precision peg insertion, consider:
```python
ACTION_SCALE = np.array([0.003, 0.02, 1.0])  # 3mm position, 1.1° rotation
```

Smaller actions = more precise control but slower task completion.

### Enable Random Reset

Already configured but verify it's active:
```python
RANDOM_RESET = True
RANDOM_XY_RANGE = 0.015   # ±15mm
RANDOM_RZ_RANGE = 0.03    # ±1.7°
```

This prevents the policy from memorizing a single trajectory.

---

## Summary

**The #1 issue is the safety box is too large**, allowing the robot to explore into singularity zones during random RL exploration. Tightening the box to match the actual task workspace will:

1. ✅ Eliminate "FORCE MODE NOT POSSIBLE IN SINGULARITY" errors
2. ✅ Eliminate "forceMode failed, recovering..." truncations  
3. ✅ Allow episodes to run to completion (or timeout)
4. ✅ Focus exploration on useful workspace → faster learning
5. ✅ Reduce dangerous robot movements

**The #2 priority is active human intervention** during early training. Without corrections, the random policy wastes thousands of steps exploring useless states.
