# 🔩 Peg-in-the-Hole Task — Complete Pipeline

> **Goal:** Collect 5 teleoperated demos → Fine-tune π0 with LoRA → Deploy on UR5e

---

## Hardware Setup

| Component | Details |
|-----------|--------|
| Robot | UR5e (IP: 172.22.1.139) |
| Gripper | Robotiq Hand-E |
| Wrist Camera | Intel RealSense D435 (serial: 034422070605) |
| Overhead Camera | Kinect v2 Xbox (serial: 000631452147) |
| GPU | NVIDIA RTX 5070 Ti (16GB VRAM) |
| Teleop | Keyboard (cartesian mode) |

---

## Complete Pipeline (All Steps)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Record demos        → 4-5 episodes of peg insertion       │
│  Step 2: Symlink dataset     → link to HF cache (one-time)         │
│  Step 3: Compute norm stats  → normalize data for training         │
│  Step 4: Train π0 LoRA       → fine-tune on your demos (~2-4 hrs)  │
│  Step 5: Serve model         → start policy server                 │
│  Step 6: Run on robot        → autonomous peg insertion!           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Record Demonstrations

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
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --dataset.episode_time_s=100 \
    --dataset.push_to_hub=False
```

> **Notes:**
> - Without `--resume`: old dataset is deleted and a fresh one is created.
> - With `--resume`: appends to existing dataset (only works if dataset was created with the SAME lerobot version).
> - If you get `FileNotFoundError: meta/episodes` with `--resume`, start fresh without it.

### Recording Key Controls

| Key | Action |
|-----|--------|
| **SPACE** | Start recording (begins saving frames) |
| **→ (Right)** | End episode (save and move to next) |
| **← (Left)** | Discard episode (re-record) |
| **ESC** | Stop all recording and exit |
| **G** | Toggle gripper (open ↔ close) |
| **W/S** | Move away/toward (±X) |
| **A/D** | Move left/right (±Y) |
| **Q/E** | Move up/down (±Z) |
| **I/K** | Rotate CW/ACW around X |
| **J/L** | Rotate CW/ACW around Y |
| **U/O** | Rotate CW/ACW around Z |
| **+/−** | Speed multiplier |

### Recording Workflow (per episode)

```
  ○ WAITING ─────────────────────────────────────────────
  │  Robot moves via teleop but NOT recording.
  │  Position robot at start (above peg).
  │
  │  Press SPACE
  ▼
  ● RECORDING ───────────────────────────────────────────
  │  Frames saved at 30 FPS. Perform the task:
  │    1. Position above peg (W/A/S/D)
  │    2. Lower to grasp (E)
  │    3. Close gripper (G)
  │    4. Lift peg (Q)
  │    5. Position above hole (W/A/S/D)
  │    6. Insert peg (E)
  │    7. Open gripper (G) — optional
  │
  │  Press → (Right Arrow)
  ▼
  ✓ SAVED → returns to WAITING for next episode
```

**Status: ✅ DONE — 4 episodes collected**

---

## Step 2: Symlink Dataset to HuggingFace Cache (One-Time)

The training script looks for datasets in `~/.cache/huggingface/lerobot/`. Create a symlink:

```bash
mkdir -p ~/.cache/huggingface/lerobot/saifi/
ln -sf /home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual
```

**Status: ✅ DONE**

---

## Step 3: Compute Normalization Statistics (One-Time)

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

uv run scripts/compute_norm_stats.py --config-name=pi0_ur5e_peg_insertion_lora
```

Saves stats to: `assets/pi0_ur5e_peg_insertion_lora/saifi/ur5e-peg-insertion-dual/`

**Status: ✅ DONE**

---

## Step 4: Train π0-FAST with LoRA (LOCAL GPU)

> ⚠️ **RTX 5070 Ti (16GB):** Even π0-FAST needs memory optimization.
> Use `scripts/train_local.sh` which sets XLA env vars to maximize GPU memory.

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

# USE THIS SCRIPT (sets XLA memory flags automatically)
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_run1 --overwrite
```

> **Flags:**
> - `--overwrite` : restart training from scratch (removes old checkpoints)
> - `--resume` : continue training from last saved checkpoint (bare flag, no =True)
> - `--exp-name=NAME` : required, names the checkpoint directory

### Available Training Configs for Peg Insertion

| Config | Model | VRAM | Where | Batch | Rank | Speed |
|--------|-------|------|-------|-------|------|-------|
| `pi0_fast_ur5e_peg_insertion_lora` | π0-FAST LoRA | **15.7 GB** | ✅ Local (16GB) | 1 | 4 | 2.1 it/s |
| `pi0_ur5e_peg_insertion_lora` | π0 LoRA | ~24 GB | HPC (V100 32GB) | 4 | 16 | — |
| `pi05_ur5e_peg_insertion_lora` | π0.5 LoRA | ~28 GB | HPC (V100 32GB) | 16 | 16 | — |

### Training Config Details (π0-FAST, Local) — TESTED & WORKING ✅

| Parameter | Value |
|-----------|-------|
| Config name | `pi0_fast_ur5e_peg_insertion_lora` |
| Model | π0-FAST (PaLI-Gemma 2B LoRA rank=4 + FAST tokenizer) |
| Dataset | `saifi/ur5e-peg-insertion-dual` |
| Action horizon | 30 steps |
| Training steps | 30,000 |
| Batch size | **1** |
| LoRA rank | **4** (saves ~3.7 GiB vs default rank=16) |
| Save interval | Every 5,000 steps |
| VRAM | ~15.7 GB / 16 GB (96% utilization) |
| Speed | 2.1 it/s (~3h 53min total) |

### What `train_local.sh` does

```bash
# These env vars are critical for fitting in 16GB:
export XLA_PYTHON_CLIENT_PREALLOCATE=true     # pre-allocate pool (avoids fragmentation!)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95    # use 95% of GPU (default 75%)
export XLA_FLAGS="--xla_gpu_autotune_level=0"  # skip autotuning (saves ~700MB)
```

### ⚠️ Critical: Kill zombie processes before training!

```bash
# Check for old python processes holding GPU memory
nvidia-smi
# If you see python processes using memory, kill them:
kill -9 <PID>
```

### Train on HPC cluster (π0 or π0.5)

See `HPC_TRAINING.md` for full HPC setup. Quick commands:

```bash
# Submit π0 LoRA job (V100 32GB)
sbatch scripts/train_hpc_pi0.sh

# Submit π0.5 LoRA job (V100 32GB)
sbatch scripts/train_hpc_pi05.sh
```

### Monitor Training

```bash
# Check checkpoints being saved
watch -n 10 "ls checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/ 2>/dev/null"

# WandB dashboard
# https://wandb.ai/saifi/openpi
```

### Checkpoints Saved At

```
checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/
├── 5000/
├── 10000/
├── 15000/
├── 20000/
├── 25000/
└── 30000/    ← final
```

---

## Step 5: Serve the Trained Model

```bash
# Terminal 1: Policy server
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/30000
```

> Replace `30000` with the checkpoint step you want. Wait until you see "Waiting for connections..."

---

## Step 6: Run Inference on Robot

```bash
# Terminal 2: Robot control
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost \
    --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --fps=30
```

> ⚠️ **SAFETY:** Keep hand on E-STOP. Press ESC to stop immediately.

---

## Quick Reference (Copy-Paste All Steps)

```bash
# ═══════════════════════════════════════════════════════════
# STEP 1: Record 5 episodes (start fresh)
# ═══════════════════════════════════════════════════════════
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate

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
    --dataset.push_to_hub=False

# ═══════════════════════════════════════════════════════════
# STEP 2: Symlink dataset (one-time)
# ═══════════════════════════════════════════════════════════
mkdir -p ~/.cache/huggingface/lerobot/saifi/
ln -sf ~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual

# ═══════════════════════════════════════════════════════════
# STEP 3: Compute norm stats (one-time per dataset)
# ═══════════════════════════════════════════════════════════
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate

uv run scripts/compute_norm_stats.py --config-name=pi0_fast_ur5e_peg_insertion_lora

# ══════════════════════════════════════════════════��════════
# STEP 4: Train π0-FAST LoRA (LOCAL GPU with memory optimization)
# ═══════════════════════════════════════════════════════════
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_run1 --overwrite

# ═══════════════════════════════════════════════════════════
# STEP 5: Serve model (Terminal 1)
# ═══════════════════════════════════════════════════════════
cd ~/ur5e_hande_workspace/openpi_ur5e/openpi-ur5e && source .venv/bin/activate

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir=./checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/30000

# ═══════════════════════════════════════════════════════════
# STEP 6: Run on robot (Terminal 2)
# ═══════════════════════════════════════════════════════════
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello && source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost \
    --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --fps=30
```

---

## Utility: Move Robot to Home Position

```bash
cd ~/ur5e_hande_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

python -c "
import rtde_control
import numpy as np
rtc = rtde_control.RTDEControlInterface('172.22.1.139')
q = np.deg2rad([33.56, -76.79, -132.20, -60.98, 90.22, 35.98]).tolist()
rtc.moveJ(q, 0.5, 0.5)
rtc.stopScript()
print('✓ Robot moved to home position')
"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `--resume=True` error | Use bare `--resume` (NOT `--resume=True`) |
| `FileNotFoundError: meta/episodes` | Old dataset format — start fresh without `--resume` |
| `FileExistsError: Checkpoint directory already exists` | Add `--overwrite` to training command |
| `Normalization stats not found` | Run `uv run scripts/compute_norm_stats.py --config-name=pi0_ur5e_peg_insertion_lora` first |
| `FileNotFoundError: .cache/huggingface/lerobot/...` | Create symlink: `ln -sf ~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual` |
| `VIRTUAL_ENV=... does not match` | Harmless warning. Or run `deactivate` first, then `source .venv/bin/activate` in the correct project dir |
| `Calibration failed: STOPPED_INNER_OBJECT` | Clear protective stop on UR teach pendant → Power ON → START → Remote Control mode |
| SPACE key conflict (gripper + record) | Fixed: **G** = gripper, **SPACE** = start recording |
| Robot not connecting (3 retries) | Check: pendant clear, Remote Control mode, `ping 172.22.1.139` |
| `RepositoryNotFoundError` on Hub | Dataset is local-only — create the symlink (Step 2) |
| VRAM OOM during training | 1) Kill zombie processes: `nvidia-smi` then `kill -9 <PID>`. 2) Use `pi0_fast_ur5e_peg_insertion_lora` (rank=4, batch=1). 3) Must have ~15.4 GB free before starting. |
| Training too slow | Normal: 30k steps takes 2-4 hrs on RTX 5070 Ti |

---

## Expected Results with 4-5 Episodes

With only 4-5 demonstrations, expect:
- ✅ Robot moves toward the peg area
- ✅ Attempts grasping motion
- ⚠️ May not complete full insertion reliably
- ⚠️ Position accuracy will be limited

This is a **pipeline validation** — proves the full system works end-to-end. For reliable task completion, collect 20-50 episodes.

---

## Dataset Location

```
~/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual/
├── data/chunk-000/file-000.parquet    # State + action data
├── meta/info.json                      # Dataset metadata
├── meta/stats.json                     # Dataset statistics
├── videos/observation.images.wrist_cam/     # Wrist camera videos
└── videos/observation.images.overview_cam/  # Overhead camera videos
```

Symlinked to: `~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual`

---

## File Locations

| What | Path |
|------|------|
| Recording script | `lerobot_ur5e_gello/scripts/record.py` |
| Keyboard teleop | `lerobot_ur5e_gello/lerobot_teleoperator_keyboard_ur5e/.../keyboard_ur5e.py` |
| Robot driver | `lerobot_ur5e_gello/lerobot_robot_ur5e/.../ur5e.py` |
| Dataset | `datasets/saifi/ur5e-peg-insertion-dual/` |
| Training config | `openpi-ur5e/src/openpi/training/config.py` (search: `pi0_ur5e_peg_insertion_lora`) |
| Norm stats | `openpi-ur5e/assets/pi0_ur5e_peg_insertion_lora/saifi/ur5e-peg-insertion-dual/` |
| Checkpoints | `openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/` |
| Serve script | `openpi-ur5e/scripts/serve_policy.py` |
| Inference script | `lerobot_ur5e_gello/scripts/remote_pi_inference_dual_cam.py` |
| This README | `PEG_INSERTION_TASK.md` |
