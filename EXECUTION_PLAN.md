# RLT Execution Plan — Peg Insertion Task

## Current Status (Verified ✅)

| Component | Status | Details |
|-----------|--------|--------|
| GPU | ✅ | RTX 5070 Ti, PyTorch 2.12+cu128, JAX 0.6.2 |
| RL Token Model | ✅ | 128M params, loss=0.109, token_dim=512 |
| Embeddings | ✅ | 199 frames × [948, 2048] from real demos |
| z_rl extraction | ✅ | 12.9ms/frame on GPU |
| RLT Buffer | ✅ | Chunked transitions (C=10) |
| VLA Checkpoint | ✅ | π0-FAST LoRA, 30k steps |
| Unit Tests | ✅ | 23/23 pass (14 rl_token + 9 buffer) |
| LeRobot Dataset | ✅ | 4 episodes, 199 frames (existing) |

---

## Architecture (from paper)

```
┌─────────────────────────────────────────────────────────────────────┐
│ VLA (π0-FAST / π0 / π0.5) — FROZEN after fine-tuning               │
│ Input: images + language → ã_ref (action chunk, C=10×6D)           │
│ Internal: VLM embeddings [948, 2048]                                │
└────────────┬───────────────────────────────────────┬────────────────┘
             │ ã_ref (10×6)                          │ embeddings
             │                                       ▼
             │                        ┌─────────────────────────────┐
             │                        │ RL Token Encoder (frozen)   │
             │                        │ [948, 2048] → z_rl [512]    │
             │                        └──────────────┬──────────────┘
             │                                       │ z_rl
             ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SAC Actor (RLPD) — TRAINED online                                   │
│ Input: [z_rl(512) | proprio(19) | ã_ref(10×6)]                     │
│ Output: residual_chunk (10×6), clipped ±max_residual                │
│ Final: action = ã_ref + residual                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: Validate with 4 Demos (DONE ✅)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
PY=ur5e_hil_serl/.venv/bin/python

# Run all tests
$PY rlt/tests/test_rl_token.py   # 14 pass
$PY rlt/tests/test_buffer.py     # 9 pass

# Re-train RL Token (if needed, ~8 min GPU)
$PY -m rlt.training.train_rl_token \
  --cache checkpoints/rl_token/embeddings_peg_insertion_real.pt \
  --save_path checkpoints/rl_token/peg_insertion_real_v1.pt \
  --token_dim 512 --enc_layers 2 --dec_layers 2 \
  --steps 2000 --batch_size 32 --lr 3e-4 --device cuda

# Re-extract embeddings (requires VLA loaded, ~5 min)
$PY -m rlt.training.extract_embeddings_real \
  --vla_ckpt openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999 \
  --vla_config pi0_fast_ur5e_peg_insertion_lora \
  --demo_root openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
  --save_path checkpoints/rl_token/embeddings_peg_insertion_real.pt
```

---

## PHASE 2: Collect 50 Demos

### Demo Requirements (compatible with π0, π0-FAST, π0.5):

| Field | Value |
|-------|-------|
| Format | LeRobot v2.1 (parquet + mp4) |
| Episodes | 50+ |
| Cameras | wrist_1 (640×480) + overview (640×480) |
| State | 7D joint pos OR 19D (tcp+vel+force+gripper) |
| Action | 6D tcp delta (matching existing SERL setup) |
| FPS | 10 Hz |
| Task | "Pick up the peg and insert it into the hole" |

### Collection:

```bash
cd /home/robolab-2/ur5e_hande_workspace/ur5e_hande_lerobot
source .venv/bin/activate

python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/ur5e_hande.yaml \
  --fps 10 \
  --root outputs/train/ur5e_peg_insertion_50ep \
  --repo-id robolab/ur5e_peg_insertion_50ep \
  --num-episodes 50 \
  --warmup-time-s 2 \
  --episode-time-s 30 \
  --reset-time-s 10

# Verify
python -c "import pandas as pd; df=pd.read_parquet('outputs/train/ur5e_peg_insertion_50ep/data/train-00000-of-00001.parquet'); print(f'Frames: {len(df)}, Episodes: {df.episode_index.nunique()}')"

# Push to HF (for HPC)
python lerobot/scripts/push_dataset_to_hub.py \
  --root outputs/train/ur5e_peg_insertion_50ep \
  --repo-id robolab/ur5e_peg_insertion_50ep
```

---

## PHASE 3: VLA Fine-tuning

### 3A: π0-FAST (Local — RTX 5070 Ti)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e

# Config already exists: pi0_fast_ur5e_peg_insertion_lora
# Update data path in src/openpi/training/config.py if needed

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run openpi_train pi0_fast_ur5e_peg_insertion_lora
# ~5 hours, batch_size=2, 30k steps
```

### 3B: π0 (HPC Cluster)

```bash
# Add config "pi0_ur5e_peg_insertion_lora" in config.py
# Same data, different base model
uv run openpi_train pi0_ur5e_peg_insertion_lora
```

### 3C: π0.5 (HPC Cluster — needs A100 80GB)

```bash
# Add config "pi05_ur5e_peg_insertion_lora"
uv run openpi_train pi05_ur5e_peg_insertion_lora
```

---

## PHASE 4: RL Token Training (per VLA)

Each VLA produces different embeddings → separate RL Token models.

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
PY=ur5e_hil_serl/.venv/bin/python

# For π0-FAST (already done with 4 demos, redo with 50):
$PY -m rlt.training.extract_embeddings_real \
  --vla_ckpt openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/NEW_RUN/29999 \
  --vla_config pi0_fast_ur5e_peg_insertion_lora \
  --demo_root /path/to/ur5e_peg_insertion_50ep \
  --save_path checkpoints/rl_token/embeddings_pi0fast_50ep.pt

$PY -m rlt.training.train_rl_token \
  --cache checkpoints/rl_token/embeddings_pi0fast_50ep.pt \
  --save_path checkpoints/rl_token/pi0fast_50ep_v1.pt \
  --token_dim 512 --steps 5000 --batch_size 32 --device cuda

# Same for π0 and π0.5 (different checkpoint paths)
```

---

## PHASE 5: Online RL (Robot Required)

### Terminal 1: VLA Server

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
uv run openpi_server --config pi0_fast_ur5e_peg_insertion_lora
```

### Terminal 2: Online RL Training

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
PY=ur5e_hil_serl/.venv/bin/python
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PYTHONPATH"

$PY -m rlt.examples.peg_insertion.train_rlt \
  --vla_url ws://localhost:8000

# Or test mode without hardware:
$PY -m rlt.examples.peg_insertion.train_rlt --fake_env
```

---

## PHASE 6: Evaluation

```bash
# VLA + RLT (full system)
$PY -m rlt.examples.peg_insertion.train_rlt --eval_only \
  --checkpoint checkpoints/rlt_runs/peg_insertion/best.pkl

# VLA-only baseline (no residual)
$PY -m rlt.examples.peg_insertion.train_rlt --eval_only --no_residual

# SERL baseline (no VLA)
cd ur5e_hil_serl && bash examples/async_peg_insert_drq/run_actor.sh
```

---

## PHASE 7: Comparison

| Method | VLA | RL | Expected Success |
|--------|-----|-----|------------------|
| VLA-only | π0-FAST | ❌ | ~40-60% |
| VLA + RLT | π0-FAST | SAC | ~80-95% |
| VLA-only | π0 | ❌ | ~50-70% |
| VLA + RLT | π0 | SAC | ~85-95% |
| VLA-only | π0.5 | ❌ | ~55-75% |
| VLA + RLT | π0.5 | SAC | ~90-98% |
| SERL (DRQ) | ❌ | DRQ | ~70-85% |

---

## Key Files

```
rlt_ur5e/
├── rlt/
│   ├── models/rl_token.py          ✅ Encoder-decoder (128M params)
│   ├── models/pi05_hook.py         ✅ VLA embedding extraction
│   ├── agents/rlt_buffer.py        ✅ Chunked replay buffer
│   ├── envs/ur5e_rlt_env.py        ✅ Gym wrapper (VLA+RL+robot)
│   ├── training/train_rl_token.py  ✅ Offline RL Token training
│   ├── training/extract_embeddings_real.py  ✅ VLA → embeddings
│   ├── examples/peg_insertion/
│   │   ├── config.py               ✅ Full config (hardware+VLA+RL)
│   │   └── train_rlt.py            ✅ Online training loop
│   └── tests/                      ✅ 23 tests all pass
├── checkpoints/rl_token/
│   ├── peg_insertion_real_v1.pt    ✅ Trained model
│   └── embeddings_peg_insertion_real.pt  ✅ 199×[948,2048]
├── openpi_ur5e/openpi-ur5e/
│   └── checkpoints/pi0_fast_ur5e_peg_insertion_lora/  ✅ 30k steps
└── ur5e_hil_serl/
    └── examples/async_peg_insert_drq/  ✅ SERL baseline
```

---

## Immediate Next Steps

1. **Collect 50 demos** with LeRobot (keyboard teleop)
2. **Re-fine-tune π0-FAST** locally with 50 demos
3. **Re-extract embeddings** and **re-train RL Token**
4. **Wire SAC** into `train_rlt.py` (the RLPD agent from ur5e_hil_serl)
5. **Online RL** on robot
6. **Repeat** for π0 / π0.5 on HPC
7. **Compare** all methods
8. **Write documentation** (replace incomplete README)
