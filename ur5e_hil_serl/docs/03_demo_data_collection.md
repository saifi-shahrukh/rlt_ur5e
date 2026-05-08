# 03 — Demonstration Data Collection

## Overview

HIL-SERL requires a small number of human demonstrations (typically 20) to bootstrap the RL policy. The human teloperates the robot to complete the task successfully.

## Prerequisites

- Robot powered on, no protective stops
- Gripper has the peg pre-grasped (for peg insertion)
- Cameras connected and working
- Terminal has focus (for keyboard control)

## Running Demo Collection

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python record_demos.py --exp_name peg_insertion --successes_needed 20
```

## Controls (FakeSpaceMouse / Keyboard)

| Key | Action |
|-----|--------|
| Arrow Up | Move +X |
| Arrow Down | Move -X |
| Arrow Left | Move -Y |
| Arrow Right | Move +Y |
| `1` | Move +Z (up) |
| `0` | Move -Z (down) |
| Right Ctrl | Toggle gripper open/close |

**Important**: The terminal window running the script must have focus for keyboard input to work (pynput requirement).

## Workflow

1. Robot starts at RESET_Q position
2. Use keyboard to guide the peg toward the hole
3. When peg is fully inserted → reward classifier detects success OR episode times out
4. Robot automatically resets to start position
5. Repeat until `successes_needed` episodes collected

## Output

Saved to: `./demo_data/peg_insertion_<N>_transitions_<timestamp>.pkl`

The pkl file contains a list of transition dictionaries:
```python
{
    'observations': {...},     # dict with images + proprio
    'actions': np.array,       # [dx, dy, dz, drx, dry, drz, grip]
    'next_observations': {...},
    'rewards': float,
    'dones': bool,
    'masks': float,
}
```

## Tips for Good Demonstrations

1. **Be consistent** — Use similar trajectories each time
2. **Be smooth** — Avoid jerky movements
3. **Vary start slightly** — Don't always follow the exact same path
4. **Include the insertion** — Make sure the peg goes fully in
5. **20 demos is typically enough** — More isn't always better

## Recording More Demos (Augmenting)

If you already have demos and want to add more:
```bash
python record_demos.py --exp_name peg_insertion --successes_needed 10
```

You can later pass multiple demo files to the training script:
```bash
python train_rlpd.py --exp_name peg_insertion \
    --demo_path ./demo_data/file1.pkl \
    --demo_path ./demo_data/file2.pkl \
    --learner
```
