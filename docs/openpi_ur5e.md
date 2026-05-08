# OpenPI-UR5e: VLA Fine-tuning & Serving

## Purpose
Fine-tunes Vision-Language-Action models (π0-FAST, π0, π0.5) on teleoperated demonstrations, then serves them as a WebSocket policy server for real-time robot control.

## Architecture
```
openpi_ur5e/
├── openpi-ur5e/                    ← Fork of Physical Intelligence's OpenPI
│   ├── src/openpi/
│   │   ├── models/                 ← π0, π0-FAST, π0.5 model definitions
│   │   ├── training/
│   │   │   ├── config.py           ← All training configs (LoRA, batch, etc.)
│   │   │   └── utils.py            ← TrainState, optimizer setup
│   │   └── policies/               ← Policy inference (action denormalization)
│   ├── scripts/
│   │   ├── train.py                ← VLA fine-tuning script
│   │   ├── serve_policy.py         ← WebSocket policy server
│   │   └── compute_norm_stats.py   ← Dataset normalization
│   ├── checkpoints/                ← Saved fine-tuned models
│   │   ��── pi0_fast_ur5e_peg_insertion_lora/
│   │       ├── peg_insertion_run1/29999/   ← 4-demo checkpoint (old)
│   │       └── peg_insertion_9demos/       ← 9-demo checkpoint (current)
│   └── assets/                     ← Normalization stats per dataset
└── lerobot_ur5e_gello/             ← Demo collection & VLA-only inference
    ├── scripts/
    │   ├── record.py               ← Record demos with keyboard teleop
    │   └── remote_pi_inference_dual_cam.py  ← VLA-only robot control
    ├── lerobot_robot_ur5e/         ← UR5e robot plugin for LeRobot
    ├── lerobot_camera_kinect/      ← Kinect v2 camera plugin
    └── lerobot_teleoperator_keyboard_ur5e/  ← Keyboard teleop plugin
```

## VLA Models Available

| Model | Config Name | VRAM | Where |
|-------|-------------|------|-------|
| π0-FAST LoRA (rank=4) | `pi0_fast_ur5e_peg_insertion_lora` | 15.7 GB | Local RTX 5070 Ti |
| π0 LoRA (rank=16) | `pi0_ur5e_peg_insertion_lora` | ~24 GB | HPC V100 32GB |
| π0.5 LoRA (rank=16) | `pi05_ur5e_peg_insertion_lora` | ~28 GB | HPC V100 32GB |

## Key Commands

```bash
# Fine-tune (local)
cd openpi_ur5e/openpi-ur5e
.venv/bin/python scripts/train.py pi0_fast_ur5e_peg_insertion_lora --exp-name=NAME --overwrite

# Serve
.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=CONFIG --policy.dir=CHECKPOINT_DIR

# Compute norm stats
.venv/bin/python scripts/compute_norm_stats.py --config-name=CONFIG
```

## Venv
- **Path:** `openpi_ur5e/openpi-ur5e/.venv/`
- **Python:** 3.11
- **Key packages:** JAX, Flax, optax, orbax, openpi (editable)
