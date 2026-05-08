# RLT-UR5e: Reinforcement Learning Tokens for Peg Insertion

> Implementation of ["RL Token: Bootstrapping Online RL with VLAs"](https://arxiv.org/abs/2604.23073) (Physical Intelligence, 2026)
> on UR5e + Robotiq Hand-E for precision peg insertion.

---

## Overview

This project fine-tunes Vision-Language-Action (VLA) models and then improves their precision using online Reinforcement Learning via the RL Token framework.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Fine-tune VLA (π0-FAST) on teleoperated demonstrations       │
│  Phase 2: Train RL Token encoder (compress VLA embeddings → 512D)      │
│  Phase 3: Online RL — SAC learns residual corrections on real robot    │
│  Phase 4: Compare π0-FAST, π0, π0.5 with/without RLT                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware

| Component | Details |
|-----------|--------|
| Robot | Universal Robots UR5e (IP: 172.22.1.139) |
| Gripper | Robotiq Hand-E |
| Wrist Camera | Intel RealSense D435 (serial: 034422070605) |
| Overhead Camera | Kinect v2 Xbox (serial: 000631452147) |
| GPU | NVIDIA RTX 5070 Ti (16GB VRAM) |
| Teleop | Keyboard (cartesian mode) |

## Project Structure

```
rlt_ur5e/
├── COMMANDS.md                  ← Full command reference
├── README.md                    ← This file
├── rlt.pdf                      ← Original paper
├── scripts/
│   ├── start_vla_server.sh      ← Terminal 1: serve VLA policy
│   ├── run_vla_only.sh          ← VLA-only baseline inference
│   ├── run_rlt_training.sh      ← Full RLT online RL training
│   ├── run_rlt_eval.sh          ← Evaluate saved checkpoints
│   └── collect_50_demos.sh      ← Record 50 demonstrations
├── rlt/                         ← RLT implementation
│   ├── agents/
│   │   ├── sac_agent.py         ← JAX SAC (TD3-style, 2 Q-funcs)
│   │   └── rlt_buffer.py        ← Replay buffer (chunked)
│   ├── models/
│   │   ├── rl_token.py          ← RL Token encoder-decoder
│   │   ├── train_rl_token.py    ← RL Token training script
│   │   └── extract_embeddings.py← Extract VLA embeddings
│   ├── envs/
│   │   └── ur5e_rlt_env.py      ← Gym env (SERL + VLA + RL Token)
│   └── examples/
│       └── peg_insertion/
│           ├── config.py        ← All hyperparameters
│           └── train_rlt.py     ← Main training loop
├── checkpoints/
│   ├── rl_token/                ← Trained RL Token models
│   └── rlt_runs/                ← SAC agent checkpoints
├── openpi_ur5e/                 ← VLA fine-tuning & serving
│   ├── openpi-ur5e/             ← OpenPI fork (π0/π0-FAST/π0.5)
│   └── lerobot_ur5e_gello/      ← Demo collection & inference
└── ur5e_hil_serl/               ← Robot control & reward classifier
```

## Virtual Environments (3 venvs)

| Venv | Python | Purpose |
|------|--------|---------|
| `ur5e_hil_serl/.venv/` | 3.10 | RLT training, SAC, RL Token, robot control |
| `openpi_ur5e/openpi-ur5e/.venv/` | 3.11 | VLA server (serve_policy.py) |
| `openpi_ur5e/lerobot_ur5e_gello/.venv/` | 3.11 | Demo collection, VLA-only inference |

## Quick Start

### 1. Start VLA Server (Terminal 1)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e

.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_fast_ur5e_peg_insertion_lora \
  --policy.dir checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999
```

### 2. VLA-Only Baseline (Terminal 2)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

python scripts/remote_pi_inference_dual_cam.py \
  --ip=localhost --port=8000 \
  --prompt="Pick up the peg and insert it into the hole." \
  --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --fps=30
```

### 3. RLT Online RL (Terminal 2)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"
export JAX_PLATFORMS=cpu

python -m rlt.examples.peg_insertion.train_rlt
```

### 4. Collect 50 Demos

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

python scripts/record.py \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 \
    --teleop.type=keyboard_ur5e --teleop.robot_ip=172.22.1.139 --teleop.mode=cartesian \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.root=/home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets \
    --dataset.num_episodes=50 --dataset.fps=30 --dataset.push_to_hub=False
```

## Architecture (from Paper)

```
Images + Language → [FROZEN VLA] → embeddings → [RL Token Encoder] → z_rl (512D)
                        ↓                                               ↓
                   ã_ref (10×6D)                                   z_rl (512D)
                        ↓                                               ↓
                        └────────────────────┬──────────────────────────┘
                                             ↓
                  [SAC Actor: (z_rl, proprio, ã_ref) → residual chunk]
                  [SAC Critic: Q(state, action) → value]
                                             ↓
                  final_action = ã_ref + clip(residual, ±3mm/±1.1°)
                                             ↓
                  [Robot executes 10 steps open-loop]
                                             ↓
                  [Reward: image classifier detects success → +1]
```

**Key:** SAC sees NO images — only z_rl (512D compressed VLA state) + proprio (19D) + reference chunk (60D).

## Reward

We use a **trained image reward classifier** (from HIL-SERL):
- Watches wrist + overview cameras
- 3 consecutive frames with probability > 0.70 → reward = +1
- Located at: `ur5e_hil_serl/examples/classifier_ckpt/checkpoint_150/`

## Current Status

- [x] VLA fine-tuned (π0-FAST, 4 demos, 30k steps)
- [x] RL Token trained (loss=0.109, 512D)
- [x] SAC agent wired (TD3-style, 2 Q-funcs, [256,256])
- [x] Full pipeline verified (fake env)
- [x] VLA server working
- [ ] VLA-only baseline measured (need clear gripper)
- [ ] RLT online RL on real robot
- [ ] 50 demos collected
- [ ] Re-train with 50 demos
- [ ] π0/π0.5 comparison (HPC)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module 'lerobot'` | Venv broken. Fix `pyvenv.cfg` and `activate` paths (see COMMANDS.md) |
| `STOPPED_INNER_OBJECT` | Remove peg from gripper → clear protective stop → Remote Control mode |
| `openpi_server: not found` | Use `.venv/bin/python scripts/serve_policy.py` instead |
| BLAS error (JAX) | Set `export JAX_PLATFORMS=cpu` |
| VRAM OOM | Kill zombie: `nvidia-smi` → `kill -9 <PID>` |

## References

- [RL Token Paper](https://arxiv.org/abs/2604.23073)
- [π0 / OpenPI](https://github.com/Physical-Intelligence/openpi)
- [HIL-SERL](https://github.com/rail-berkeley/serl)
- [LeRobot](https://github.com/huggingface/lerobot)
