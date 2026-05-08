# OpenPI UR5e — Fine-Tuning & Deployment Pipeline for Robot Manipulation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/framework-JAX-red.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end pipeline for **fine-tuning π0/π0-FAST vision-language-action models** on a **UR5e robot** with dual cameras, then deploying for real-world manipulation tasks (e.g., peg-in-hole insertion).

---

## 🎯 What This Repo Does

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  1. RECORD      │ ──► │  2. TRAIN        │ ──► │  3. SERVE       │ ──► │  4. DEPLOY       │
│  Demonstrations │     │  Fine-tune π0    │     │  Policy Server  │     │  Robot Inference │
│  (teleoperate)  │     │  LoRA on GPU     │     │  (WebSocket)    │     │  (30Hz control)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
```

| Step | What Happens | Tools |
|------|-------------|-------|
| **Record** | Teleoperate the UR5e via keyboard/GELLO, record state+actions+images | LeRobot + RealSense + Kinect |
| **Train** | Fine-tune π0-FAST (16GB GPU) or π0/π0.5 (32GB HPC) with LoRA | JAX + OpenPI + Orbax |
| **Serve** | Load checkpoint, start WebSocket server on port 8000 | OpenPI serve_policy.py |
| **Deploy** | Connect to robot, stream images → get actions → execute at 30Hz | OpenPI client + ur_rtde |

---

## 📁 Repository Structure

```
openpi_ur5e/
├── README.md                          ← You are here
├── openpi-ur5e/                       ← Model training & serving (JAX/Flax)
│   ├── src/openpi/                    ← Core model code (π0, π0-FAST, LoRA)
│   ├── scripts/
│   │   ├── train.py                   ← Main training script
│   │   ├── train_local.sh             ← Local GPU wrapper (XLA memory flags)
│   │   ├── train_hpc_pi0.sh           ← SLURM job for π0 on HPC
│   │   ├── train_hpc_pi05.sh          ← SLURM job for π0.5 on HPC
│   │   ├── serve_policy.py            ← Start WebSocket policy server
│   │   └── compute_norm_stats.py      ← Compute dataset normalization
│   ├── checkpoints/                   ← Saved model checkpoints (gitignored)
│   └── assets/                        ← Normalization statistics per config
│
├── lerobot_ur5e_gello/                ← Robot control & data collection
│   ├── scripts/
│   │   ├── record.py                  ← Record demonstrations (SPACE to start)
│   │   ├── remote_pi_inference_dual_cam.py  ← Run policy on robot (dual camera)
│   │   ├── teleoperate.py             ← Manual teleoperation
│   │   └── kinect_zmq_server.py       ← Stream Kinect over ZMQ
│   ├── lerobot_robot_ur5e/            ← UR5e robot driver (ur_rtde)
│   ├── lerobot_camera_kinect/         ← Azure Kinect v2 camera driver
│   ├── lerobot_camera_zmq/            ← ZMQ camera streaming
│   ├── lerobot_teleoperator_keyboard_ur5e/  ← Keyboard teleoperation
│   ├── lerobot_teleoperator_gello/    ← GELLO teleoperation device
│   └── openpi_client/                 ← WebSocket client for policy server
│
├── datasets/                          ← Recorded demonstrations (gitignored)
├── HPC_TRAINING.md                    ← HPC cluster training guide
├── PEG_INSERTION_TASK.md              ← Peg insertion task documentation
├── UR5e_OpenPI_PIPELINE_README.md     ← Detailed pipeline reference
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

| Component | Requirement |
|-----------|------------|
| Robot | UR5e with Robotiq Hand-E gripper |
| Cameras | Intel RealSense D435 (wrist) + Azure Kinect v2 (overhead) |
| GPU (local) | NVIDIA GPU with ≥16GB VRAM (RTX 4080/5070 Ti/A4000+) |
| GPU (HPC) | NVIDIA V100/A100 with ≥32GB VRAM |
| Python | 3.11 |
| OS | Ubuntu 22.04+ |

### 1. Clone & Setup

```bash
# Clone the repository
git clone git@github.com:saifi-shahrukh/openpi_ur5e-.git
cd openpi_ur5e-

# Setup openpi-ur5e (training & serving)
cd openpi-ur5e
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv sync  # Creates .venv and installs all dependencies
cd ..

# Setup lerobot_ur5e_gello (robot control)
cd lerobot_ur5e_gello
uv sync  # Creates .venv and installs all dependencies
cd ..
```

### 2. Record Demonstrations

```bash
cd lerobot_ur5e_gello && source .venv/bin/activate

# Record 30 episodes of peg insertion
python scripts/record.py \
  --robot.type=ur5e_dual_cam \
  --robot.ip=172.22.1.139 \
  --fps=30 \
  --repo-id=saifi/ur5e-peg-insertion-dual \
  --num-episodes=30 \
  --push-to-hub=0
```

> **Controls:** SPACE = start/stop recording, G = toggle gripper, Arrow keys = move robot

### 3. Compute Normalization Statistics

```bash
cd openpi-ur5e && source .venv/bin/activate

uv run scripts/compute_norm_stats.py --config-name=pi0_fast_ur5e_peg_insertion_lora
```

### 4. Train the Model

```bash
# Local training (16GB GPU — π0-FAST with LoRA rank=4)
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora \
  --exp-name=peg_insertion_run1 \
  --overwrite

# Resume from checkpoint
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora \
  --exp-name=peg_insertion_run1 \
  --resume
```

### 5. Serve the Trained Model

```bash
cd openpi-ur5e && source .venv/bin/activate

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_ur5e_peg_insertion_lora \
  --policy.dir=./checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999
```

> Wait for: `server listening on 0.0.0.0:8000`

### 6. Run on Robot

```bash
# In a NEW terminal
cd lerobot_ur5e_gello && source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
  --ip=localhost --port=8000 \
  --prompt="Pick up the peg and insert it into the hole." \
  --robot.type=ur5e_dual_cam \
  --robot.ip=172.22.1.139 \
  --fps=30
```

> ⚠️ **SAFETY:** Keep hand on E-STOP! Press ESC to stop.

---

## 🖥️ Hardware Setup

```
                    ┌──────────────────┐
                    │   Workstation    │
                    │  (GPU Training)  │
                    └────────┬─────────┘
                             │ Ethernet
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌──────▼───────┐  ┌─────▼──────┐
    │   UR5e       │  │  RealSense   │  │  Kinect v2 │
    │ 172.22.1.139 │  │  D435 Wrist  │  │  Overhead  │
    │ + Hand-E     │  │  USB 3.0     │  │  USB 3.0   │
    └──────────────┘  └──────────────┘  └────────────┘
```

---

## 🏋️ Training Configurations

| Config | GPU | LoRA Rank | Batch Size | Use Case |
|--------|-----|-----------|------------|----------|
| `pi0_fast_ur5e_peg_insertion_lora` | 16GB (local) | 4 | 1 | Quick iteration |
| `pi0_ur5e_peg_insertion_lora` | 32GB (HPC) | 32 | 16 | Best quality |
| `pi05_ur5e_peg_insertion_lora` | 32GB (HPC) | 32 | 16 | π0.5 architecture |

### Training Tips

- **Minimum 30 demonstrations** for a simple task (50+ recommended)
- Vary start positions, approach angles, and speeds in demos
- π0-FAST uses DCT tokenization → needs more data than π0
- Loss should reach <0.5 for good task performance
- Always kill zombie GPU processes before training: `pkill -f train.py`

---

## 📡 Architecture: Serve → Infer

```
┌─────────────────────────┐         WebSocket (port 8000)         ┌─────────────────────────┐
│     POLICY SERVER       │ ◄──────────────────────────────────── │     ROBOT CLIENT        │
│  (openpi-ur5e)          │                                       │  (lerobot_ur5e_gello)   │
│                         │  ┌─────────────────────────────────┐  │                         │
│  • Load π0 checkpoint   │  │  Message (per inference call):  │  │  • Read joint state     │
│  • GPU inference (JAX)  │  │  {                              │  │  • Capture 2 images     │
│  • Return action chunk  │  │    "state": [6 joints+gripper], │  │  • Send to server       │
│    (30 timesteps)       │  │    "images": {wrist, overhead}, │  │  • Execute actions      │
│                         │  │    "prompt": "..."              │  │    at 30Hz              │
│                         │  │  }                              │  │                         │
└─────────────────────────┘  └─────────────────────────────────┘  └─────────────────────────┘
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during training | Kill zombie processes: `pkill -f python.*train` |
| RTDE registers in use | Kill old connections: `pkill -f rtde` or restart robot |
| Token decode errors | Model undertrained — need more demos + more steps |
| `--resume=True` error | Use bare `--resume` flag (no `=True`) |
| Slow first inference (~13s) | Normal — JAX JIT compilation. Subsequent: ~1.5s |
| Robot protective stop | Clear on teach pendant, ensure Remote Control mode |

---

## 📚 Related Documentation

| Document | Description |
|----------|-------------|
| [PEG_INSERTION_TASK.md](PEG_INSERTION_TASK.md) | Task-specific setup and training guide |
| [HPC_TRAINING.md](HPC_TRAINING.md) | GPU cluster (SLURM + Enroot) training guide |
| [UR5e_OpenPI_PIPELINE_README.md](UR5e_OpenPI_PIPELINE_README.md) | Full pipeline reference |
| [openpi-ur5e/LOCAL_TRAINING.md](openpi-ur5e/LOCAL_TRAINING.md) | Local GPU training guide |

---

## 🛠️ Development

### Adding a New Task

1. Record demonstrations with a new `--repo-id`
2. Add a new config in `openpi-ur5e/src/openpi/training/config.py`
3. Compute normalization stats
4. Train and evaluate

### Project Dependencies

- **openpi-ur5e:** JAX, Flax, Orbax, PaliGemma, SigLIP, draccus
- **lerobot_ur5e_gello:** ur_rtde, pyrealsense2, pyk4a, LeRobot, websockets

---

## 📖 References

1. **Physical Intelligence π0:** [Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", 2024](https://arxiv.org/abs/2410.24164)
2. **π0-FAST:** [Pertsch et al., "Fast Action Tokenization for Robotics", 2025](https://arxiv.org/abs/2501.09747)
3. **OpenPI (open-source π0):** [GitHub — Physical Intelligence](https://github.com/Physical-Intelligence/openpi)
4. **LeRobot:** [GitHub — Hugging Face LeRobot](https://github.com/huggingface/lerobot)
5. **UR RTDE:** [Universal Robots RTDE Interface](https://sdurobotics.gitlab.io/ur_rtde/)
6. **LoRA:** [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021](https://arxiv.org/abs/2106.09685)
7. **PaliGemma:** [Google DeepMind PaliGemma VLM](https://ai.google.dev/gemma/docs/paligemma)
8. **SigLIP:** [Zhai et al., "Sigmoid Loss for Language Image Pre-Training", 2023](https://arxiv.org/abs/2303.15343)

---

## 📄 License

MIT License — See [LICENSE](openpi-ur5e/LICENSE) for details.

---

**Maintainer:** Shahrukh Saifi (shahrukh.saifi20@gmail.com)
