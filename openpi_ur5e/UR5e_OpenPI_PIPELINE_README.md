# UR5e + OpenPI: Complete Fine-Tuning & Deployment Pipeline

## Hardware Setup
| Component | Details |
|-----------|--------|
| Robot | Universal Robots UR5e (IP: 172.22.1.139) |
| Gripper | Robotiq Hand-E (port 63352) |
| Wrist Camera | Intel RealSense D435 (serial: 034422070605) |
| Overhead Camera | Kinect v2 Xbox (serial: 000631452147) |
| GPU | NVIDIA RTX 5070 Ti (16GB VRAM) |

## Camera Modes
| Mode | Cameras | Dataset repo_id | Training config |
|------|---------|----------------|------------------|
| **Single** | RealSense D435 wrist only | `saifi/ur5e-pick-objects` | `pi0_ur5e_single_cam_lora` |
| **Dual** | RealSense D435 wrist + Kinect v2 overhead | `saifi/ur5e-pick-objects-dual` | `pi0_ur5e_dual_cam_lora` |

## Available Training Configs

### Peg Insertion Task (Dual Camera)

| Config Name | Model | VRAM | Where | Batch | Status |
|------------|-------|------|-------|-------|--------|
| **`pi0_fast_ur5e_peg_insertion_lora`** | **π0-FAST LoRA (rank=4)** | **15.7 GB** | **✅ Local** | **1** | **TESTED** |
| `pi0_ur5e_peg_insertion_lora` | π0 LoRA (rank=16) | ~24 GB | HPC | 16 | Ready |
| `pi05_ur5e_peg_insertion_lora` | π0.5 LoRA (rank=16) | ~28 GB | HPC | 16 | Ready |

### General Pick/Place Configs (Dual Camera)

| Config Name | Model | Camera | VRAM |
|------------|-------|--------|------|
| `pi0_ur5e_dual_cam_lora` | π0 LoRA | Dual | ~24 GB (HPC) |
| `pi0_ur5e_dual_cam_full` | π0 Full | Dual | ~40 GB (HPC) |
| `pi0_fast_ur5e_dual_cam_lora` | π0-FAST LoRA | Dual | ~12 GB |

### General Pick/Place Configs (Single Camera)

| Config Name | Model | Camera | VRAM |
|------------|-------|--------|------|
| `pi0_ur5e_single_cam_lora` | π0 LoRA | Single | ~20 GB (HPC) |
| `pi0_ur5e_single_cam_full` | π0 Full | Single | ~40 GB (HPC) |
| `pi0_fast_ur5e_single_cam_lora` | π0-FAST LoRA | Single | ~8 GB |

> ⚠️ **RTX 5070 Ti (16GB):** Only `pi0_fast_*` configs with `rank=4, batch=1` fit locally.
> See `LOCAL_TRAINING.md` for details. For π0/π0.5, see `HPC_TRAINING.md`.

## Repository Structure
```
openpi_ur5e/
├── openpi-ur5e/                              # Model training & serving
│   ├── src/openpi/training/config.py         # ← YOUR CONFIGS HERE
│   ├── src/openpi/policies/ur5e_policy.py    # UR5e transforms
│   └── scripts/
│       ├── train.py
│       └── serve_policy.py
├── lerobot_ur5e_gello/                       # Data collection & robot control
│   ├── scripts/
│   │   ├── record_wrapper.sh                 # Wrapper (sets LD_LIBRARY_PATH)
│   │   ├── record.py                         # Recording logic
│   │   ├── remote_pi_inference_single_cam.py # Single cam inference
│   │   └── remote_pi_inference_dual_cam.py   # Dual cam inference
│   ├── lerobot_robot_ur5e/                   # Robot plugin
│   │   └── config_ur5e.py                    # UR5EConfig + UR5EDualCamConfig
│   ├── lerobot_teleoperator_keyboard_ur5e/   # Keyboard teleop plugin
│   │   ├── keyboard_ur5e.py                  # Key mapping + RTDE jog
│   │   └── ur5e_kin.py                       # Geometric Jacobian IK
│   └── lerobot_camera_kinect/                # Kinect camera plugin
├── datasets/                                 # ← RECORDED DATASETS SAVED HERE
│   └── saifi/ur5e-pick-objects-dual/
└── UR5e_OpenPI_PIPELINE_README.md            # This file
```

---

## 📂 Dataset Storage

Datasets are saved locally in the project workspace (NOT in hidden HF cache):
```
~/ur5e_hande_workspace/openpi_ur5e/datasets/<repo_id>/
```

For example:
```
~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-pick-objects-dual/
├── data/          # Parquet files with state/action data
├── meta/          # info.json, episodes.jsonl, stats
└── videos/        # Encoded MP4 videos (after consolidation)
```

- If the dataset directory already exists and `--resume` is not used, it will be **automatically removed** and started fresh (with a warning).
- To override the storage path: `--dataset.root=/path/to/custom/datasets`
- Set `--dataset.root=None` to use the default HuggingFace cache (`~/.cache/huggingface/lerobot/`).

---

## Pipeline Steps

### Step 1: Verify Config
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

python -c "
import sys; sys.path.insert(0, 'src')
from openpi.training.config import _CONFIGS_DICT
for n in sorted(_CONFIGS_DICT):
    if 'ur5e' in n:
        print(f'  {n}')
print('\n✓ All configs loaded!')
"
```

### Step 2: Record Demonstrations

#### Dual Camera Mode (RealSense wrist + Kinect v2 overhead)
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

./scripts/record_wrapper.sh \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-pick-objects-dual \
    --dataset.single_task="Pick up the green cube and lift it straight up." \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False
```

#### Single Camera Mode (RealSense wrist only)
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

./scripts/record_wrapper.sh \
    --robot.type=ur5e \
    --robot.ip=172.22.1.139 \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-pick-objects \
    --dataset.single_task="Pick up the green cube and lift it straight up." \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False
```

#### Resume an Interrupted Recording
```bash
# Add --resume to continue from where you left off (won't delete existing data)
./scripts/record_wrapper.sh \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-pick-objects-dual \
    --dataset.single_task="Pick up the green cube and lift it straight up." \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False \
    --resume
```

---

### ⌨️ Keyboard Controls (Cartesian Mode)

> All directions are from **operator's perspective** (standing in front of the robot, facing it).

**Translation:**
| Key | Action | Direction |
|-----|--------|-----------|
| **W** | Move −X | Away from you (robot pushes forward) |
| **S** | Move +X | Toward you (robot pulls back) |
| **A** | Move −Y | To your left |
| **D** | Move +Y | To your right |
| **Q** | Move +Z | Up |
| **E** | Move −Z | Down |

**Rotation** (clockwise = CW when looking at the robot along that axis):
| Key | Action | Direction |
|-----|--------|-----------|
| **I** | CW around X | Clockwise rotation around X axis |
| **K** | ACW around X | Anti-clockwise rotation around X axis |
| **J** | CW around Y | Clockwise rotation around Y axis |
| **L** | ACW around Y | Anti-clockwise rotation around Y axis |
| **U** | CW around Z | Clockwise rotation around Z axis (from above) |
| **O** | ACW around Z | Anti-clockwise rotation around Z axis (from above) |

**Controls:**
| Key | Action |
|-----|--------|
| **G** | Toggle gripper (Open ↔ Close) |
| **+/−** | Adjust speed multiplier (0.25x to 3.0x) |
| **→** (Right arrow) | End episode (save and move to next) |
| **←** (Left arrow) | Re-record (discard current episode) |
| **ESC** | Stop recording (exit completely) |

---

### 🎬 Recording Workflow (SPACE-to-Start)

The recording now has a **two-phase** workflow per episode:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: WAITING (robot moves but NO data is saved)            │
│  ─────────────────────────────────────────────────────────────  │
│  • Move robot to starting position via keyboard teleop          │
│  • Position the object/scene as needed                          │
│  • Take your time — nothing is being recorded yet               │
│                                                                 │
│  ► Press SPACE to begin recording                               │
│                                                                 │
│  Phase 2: RECORDING (frames are saved to dataset)               │
│  ─────────────────────────────────────────────────────────────  │
│  • Perform the demonstration task                               │
│  • All observations + actions saved at configured FPS           │
│                                                                 │
│  ► Press → (Right Arrow) to end & save episode                  │
│  ► Press ← (Left Arrow) to discard & re-record                  │
│                                                                 │
│  After saving: returns to Phase 1 for next episode              │
└─────────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- Previously, recording started immediately — any positioning movements were captured as part of the demo
- Now you can freely move to the start pose without polluting the training data
- This gives cleaner demonstrations with consistent start positions

**Speed Tuning:**
```bash
# Default: trans_vel=0.04 m/s, rot_vel=0.3 rad/s
# Recommended for demos:
--teleop.trans_vel=0.08 --teleop.rot_vel=0.3
# Or use +/- keys during recording to adjust speed multiplier (shown in logs)
```

**Per Episode (~5-15 seconds of recorded data):**
1. (Waiting phase) Position robot above object using W/A/D/E
2. Press **SPACE** → recording starts
3. Perform the task (grasp, lift, insert, etc.)
4. Press **→** → episode saved
5. (Automatically returns to waiting phase for next episode)

---

### Step 3: Push Dataset to Hub (optional)
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

huggingface-cli login
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    'saifi/ur5e-pick-objects-dual',
    root=Path('/home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-pick-objects-dual')
)
print(f'Episodes: {ds.num_episodes}, Frames: {ds.num_frames}')
ds.push_to_hub(tags=['ur5e', 'pick-place'], private=True)
"
```

### Step 4: Fine-Tune

#### Local GPU (RTX 5070 Ti 16GB) — π0-FAST only
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

# ⚠️  FIRST: Kill zombie processes! Check nvidia-smi
nvidia-smi  # kill any old python processes

# Train π0-FAST LoRA (rank=4, batch=1) — the ONLY config that fits 16GB
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_run1 --overwrite

# Speed: 2.1 it/s | ETA: ~3h 53min | VRAM: 15.7/16 GB
```

#### HPC Cluster (V100 32GB) — π0 and π0.5
```bash
# SSH to HPC and submit jobs (see HPC_TRAINING.md for full setup)
ssh -x saifi@hpc-headnode.iis.fhg.de
cd /data/beegfs/home/saifi/openpi-ur5e

sbatch scripts/train_hpc_pi0.sh     # π0 LoRA (batch=16)
sbatch scripts/train_hpc_pi05.sh    # π0.5 LoRA (batch=16)
```

Monitor:
```bash
# Local training
tail -f /tmp/train_peg_insertion.log

# HPC training
ssh saifi@hpc-headnode.iis.fhg.de "tail -f /data/beegfs/home/saifi/pi0_peg_*.out"

# WandB dashboard
# https://wandb.ai/saifi/openpi
```

### Step 5: Deploy on Robot

**Terminal 1 — Serve trained model:**
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

# Replace 30000 with actual checkpoint step
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_ur5e_dual_cam_lora \
    --policy.dir=./checkpoints/pi0_ur5e_dual_cam_lora/30000
```

**Terminal 2 — Run on robot (dual camera):**
```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost \
    --port=8000 \
    --prompt="Pick up the green cube and lift it straight up." \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --fps=30
```

**Or single camera:**
```bash
python scripts/remote_pi_inference_single_cam.py \
    --ip=localhost \
    --port=8000 \
    --prompt="Pick up the green cube and lift it straight up." \
    --robot.type=ur5e \
    --robot.ip=172.22.1.139 \
    --fps=30
```

⚠️ **SAFETY:** Keep hand on E-STOP. Press ESC to stop inference immediately.

---

## 🔄 Inference with Pre-Trained F-Fer Checkpoint (bolt/bearing tasks)

```bash
# Terminal 1: Serve the pre-trained tasks-merged model
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_ur_tasks_merged_lora \
    --policy.dir=./checkpoints/F-Fer/tasks-merged-lora/59999

# Terminal 2: Run inference (single cam, since F-Fer uses 3 ZED cameras → wrist duplicated)
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
python scripts/remote_pi_inference_single_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the M10 bolt and insert it into the hole until fully seated." \
    --robot.type=ur5e --robot.ip=172.22.1.139 --fps=30
```

F-Fer supported prompts:
- "Pick up the M10 bolt and insert it into the hole until fully seated."
- "Pick up the M10 bolt and insert it into the hex recess until seated."
- "Pick up the M10 bolt from the tray and put it into the blue bin."
- "Pick up the bearing and press it into the housing until flush."

---

## Multi-Object Recording Strategy

Record different objects with `--resume` to append to same dataset:

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

# Session 1: Green cube (10 episodes) — creates new dataset
./scripts/record_wrapper.sh \
    --robot.type=ur5e --robot.ip=172.22.1.139 --robot.freedrive=False \
    --teleop.type=keyboard_ur5e --teleop.robot_ip=172.22.1.139 --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-pick-objects \
    --dataset.single_task="Pick up the green cube and lift it straight up." \
    --dataset.num_episodes=10 --dataset.fps=30 --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False

# Session 2: Red sphere (10 episodes) — RESUME to append!
./scripts/record_wrapper.sh \
    --robot.type=ur5e --robot.ip=172.22.1.139 --robot.freedrive=False \
    --teleop.type=keyboard_ur5e --teleop.robot_ip=172.22.1.139 --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-pick-objects \
    --dataset.single_task="Pick up the red sphere and lift it straight up." \
    --dataset.num_episodes=10 --dataset.fps=30 --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False \
    --resume

# Session 3: Blue cuboid (10 episodes) — RESUME!
./scripts/record_wrapper.sh \
    --robot.type=ur5e --robot.ip=172.22.1.139 --robot.freedrive=False \
    --teleop.type=keyboard_ur5e --teleop.robot_ip=172.22.1.139 --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-pick-objects \
    --dataset.single_task="Pick up the blue cuboid and lift it straight up." \
    --dataset.num_episodes=10 --dataset.fps=30 --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False \
    --resume
```

The model learns to follow the language prompt to pick the correct object.

---

## Model Image Mapping

| Model Input Key | Single Camera Mode | Dual Camera Mode |
|----------------|-------------------|------------------|
| `observation/exterior_image_1_left` | wrist_cam (duplicate) | overview_cam (Kinect) |
| `observation/wrist_image_left` | wrist_cam | wrist_cam (RealSense) |
| `observation/wrist_image_right` | wrist_cam (duplicate) | wrist_cam (duplicate) |

## Data Format in Dataset (LeRobot)

Features recorded per frame:
- `observation.state` / `observation.joint_position`: 6 joint values + 1 gripper (7D float32)
- `observation.images.wrist_cam`: (480, 640, 3) uint8 RGB
- `observation.images.overview_cam`: (480, 640, 3) uint8 RGB *(dual mode only)*
- `action`: 6 joint targets + 1 gripper command (7D float32)
- `task`: string task description

---

## File Locations
| What | Where |
|------|-------|
| Training configs | `openpi-ur5e/src/openpi/training/config.py` |
| Policy transforms | `openpi-ur5e/src/openpi/policies/ur5e_policy.py` |
| Training checkpoints | `openpi-ur5e/checkpoints/<config_name>/<step>/` |
| **Recorded datasets** | **`~/ur5e_hande_workspace/openpi_ur5e/datasets/<repo_id>/`** |
| Single-cam inference | `lerobot_ur5e_gello/scripts/remote_pi_inference_single_cam.py` |
| Dual-cam inference | `lerobot_ur5e_gello/scripts/remote_pi_inference_dual_cam.py` |
| Robot config | `lerobot_ur5e_gello/lerobot_robot_ur5e/lerobot_robot_ur5e/config_ur5e.py` |
| Keyboard teleop | `lerobot_ur5e_gello/lerobot_teleoperator_keyboard_ur5e/` |
| Record wrapper | `lerobot_ur5e_gello/scripts/record_wrapper.sh` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileExistsError` on dataset | Dataset auto-clears if not using `--resume`. If you want to keep old data, use `--resume` |
| Config not found | `grep -n 'single_cam' src/openpi/training/config.py` |
| VRAM OOM during training | 1) Kill zombie processes: `nvidia-smi` → `kill -9 <PID>`. 2) Use `pi0_fast_*` with `rank=4, batch=1`. 3) Must use `./scripts/train_local.sh`. See `LOCAL_TRAINING.md`. |
| Robot not connecting | `ping 172.22.1.139`, ensure Remote Control mode on pendant |
| RealSense not found | `rs-enumerate-devices`, check USB 3.0 |
| Kinect not found | Check USB connection, verify serial in config_ur5e.py |
| pynput keyboard error | Need X11 display. Use `ssh -X` or local terminal |
| Actions too jerky | Reduce `--teleop.trans_vel` or `--teleop.rot_vel` |
| Inference too slow | Use `pi0_fast_*` config (faster tokenizer) |
| `uv run` reinstalls packages | The wrapper uses `python` directly (just activate venv first) |
| Movement direction wrong | See keyboard table: W=away, S=toward, Q=up, E=down |
| Rotation not working | Fixed: now uses geometric Jacobian (no axis-angle singularity) |
| Gripper not responding | Restart gripper via UR teach pendant → I/O → Restart |

---

## 🔩 Peg Insertion Task (Dual Camera) — Test Fine-Tuning

Collecting 4-5 episodes to validate the fine-tuning pipeline:

**Dataset path:** `~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual`

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

# Record 5 peg insertion demos (dual camera) — resume from existing 2 episodes
./scripts/record_wrapper.sh \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False \
    --resume
```

**New Recording Workflow (SPACE-to-start):**
1. Script starts → robot is live for teleop but NOT recording yet
2. Position robot at the start pose using keyboard teleop
3. Press **SPACE** → recording begins (frames saved to dataset)
4. Perform the peg insertion via keyboard teleop
5. Press **→ (Right Arrow)** → episode saved, moves to next
6. Repeat from step 2 for next episode

**Current status:** 2 episodes already collected. Need 3 more for test fine-tuning.

> ⚠️ **Note:** draccus parser requires `--resume` (not bare `--resume`). Same for all bool flags like `--dataset.push_to_hub=False`.

### Fine-tune on peg insertion data (after collecting 4-5 episodes)

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

# ⚠️ Kill zombie processes first!
nvidia-smi

# Train π0-FAST LoRA (fits on RTX 5070 Ti 16GB)
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_run1 --overwrite
# Speed: 2.1 it/s | ETA: ~3h 53min | Uses 96% of 16GB VRAM
```

> Three configs available in `src/openpi/training/config.py`:
> - `pi0_fast_ur5e_peg_insertion_lora` — ✅ Local GPU (rank=4, batch=1)
> - `pi0_ur5e_peg_insertion_lora` — HPC only (rank=16, batch=16)
> - `pi05_ur5e_peg_insertion_lora` — HPC only (rank=16, batch=16)

### Inference with trained peg insertion model

```bash
# Terminal 1: Serve the trained model
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/30000

# Terminal 2: Run on robot
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate
python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --fps=30
```

---

## Quick Start (TL;DR)

```bash
# 1. Record 20 demos with dual cameras
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate
./scripts/record_wrapper.sh \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --robot.freedrive=False \
    --teleop.type=keyboard_ur5e --teleop.robot_ip=172.22.1.139 --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.num_episodes=20 --dataset.fps=30 --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False

# 2. Create dataset symlink (one-time)
mkdir -p ~/.cache/huggingface/lerobot/saifi
ln -sf ~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual

# 3. Compute norm stats (one-time per dataset)
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate
uv run scripts/compute_norm_stats.py --config-name=pi0_fast_ur5e_peg_insertion_lora

# 4. Train (~4 hours on RTX 5070 Ti with train_local.sh)
nvidia-smi  # ⚠️ Kill any zombie python processes first!
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_run1 --overwrite

# 5. Serve trained model (Terminal 1)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/30000

# 6. Run inference on robot (Terminal 2)
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate
python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --fps=30
```

> 📖 **Detailed guides:**
> - Local training: `openpi-ur5e/LOCAL_TRAINING.md`
> - HPC training: `HPC_TRAINING.md`
> - Peg insertion pipeline: `PEG_INSERTION_TASK.md`
