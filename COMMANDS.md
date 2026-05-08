# RLT-UR5e: Complete Command Reference

> **Project:** Reinforcement Learning Tokens for UR5e Peg Insertion
> **Paper:** "RL Token: Bootstrapping Online RL with VLAs" (Physical Intelligence, 2026)

---

## Architecture Overview (from paper)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Images + Language ──→ [FROZEN VLA π0-FAST] ──→ embeddings [948, 2048]     │
│                              │                       │                    │
│                              │                       ▼                    │
│                              │            [RL Token Encoder] ──→ z_rl [512]│
│                              │                                            │
│                              ▼                       ▼                    │
│                        ã_ref (10×6D)           z_rl (512D)                │
│                              │                       │                    │
│                              └───────┬───────────────┘                    │
│                                      │                                    │
│                                      ▼                                    │
│         [SAC Actor: x=(z_rl, proprio, ã_ref) → residual a_{1:C}]         │
│         [SAC Critic: Q(x, a_{1:C}) → value]                              │
│                                      │                                    │
│                                      ▼                                    │
│                     final_action = ã_ref + residual (clipped)             │
│                                      │                                    │
│                                      ▼                                    │
│                     [Robot executes 10 steps open-loop]                   │
│                                      │                                    │
│                                      ▼                                    │
│                     [Reward: human labels success/fail]                   │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** SAC sees NO images. Images → VLA → embeddings → RL Token → z_rl (512D flat vector).
The reward is a separate binary signal (human label or classifier).

---

## Paper Hyperparameters (Appendix)

| Parameter | Value (Paper) | Our Config |
|-----------|---------------|------------|
| Actor/Critic | 2-layer MLP, hidden=256 | ✓ `[256, 256]` |
| Critic ensemble | 2 Q-functions (TD3) | ✓ `ensemble_size=2` |
| Chunk size C | 10 | ✓ `chunk_size=10` |
| Ref dropout | 50% during training | ✓ `ref_dropout=0.5` |
| BC weight β | regularize to ã | ✓ `beta=1.0` |
| Reward | Sparse binary (+1 success) | ✓ human/classifier |
| VLA chunk H | 50 steps, RL uses first 10 | ✓ |
| Actor param | Gaussian (fixed small σ) | We use learned σ (SAC) |

---

## Overall Plan (4 Phases)

```
Phase 1: VLA-only baseline (4 demos, already done)
  → Fine-tune π0-FAST ✓ → Serve → Run on robot → Measure baseline SR

Phase 2: RLT online RL (4 demos, current)
  → RL Token trained ✓ → SAC wired ✓ → Run online RL → Get improved SR  

Phase 3: Scale to 50 demos
  → Collect 50 demos → Re-fine-tune → Re-train RL Token → Better baseline + RLT

Phase 4: Compare VLAs (HPC)
  → Fine-tune π0/π0.5 on 50 demos → RL Token per VLA → Full comparison table
```

---

## Phase 1: VLA-Only Baseline (4 demos)

### Step A: Start VLA Server [Terminal 1]

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e

.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_fast_ur5e_peg_insertion_lora \
  --policy.dir checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999
```

> Wait for "Server ready" message.

### Step B: Run VLA-Only Inference [Terminal 2]

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

python scripts/remote_pi_inference_dual_cam.py \
  --ip=localhost \
  --port=8000 \
  --prompt="Pick up the peg and insert it into the hole." \
  --robot.type=ur5e_dual_cam \
  --robot.ip=172.22.1.139 \
  --fps=30
```

> Run 4-5 trials. Record success/failure manually. This is the VLA-only baseline.

---

## Phase 2: RLT Online RL (4 demos)

### Step A: Start VLA Server [Terminal 1]

Same as Phase 1 Step A above.

### Step B: Run RLT Training [Terminal 2]

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"
export JAX_PLATFORMS=cpu  # SAC on CPU (tiny MLPs), GPU for VLA+RL Token

# First: Test pipeline without hardware
python -m rlt.examples.peg_insertion.train_rlt --fake_env

# VLA-only warmup (safe, tests full hardware pipeline)
python -m rlt.examples.peg_insertion.train_rlt --warmup_only

# Full RLT training (VLA + RL Token + SAC residual learning)
python -m rlt.examples.peg_insertion.train_rlt
```

### Step C: Evaluate RLT Checkpoint [Terminal 2]

```bash
# Evaluate with RLT residual corrections
python -m rlt.examples.peg_insertion.train_rlt \
  --eval_only --eval_episodes 20 \
  --checkpoint checkpoints/rlt_runs/peg_insertion/best.pkl

# Compare: evaluate WITHOUT residual (VLA-only, same env)
python -m rlt.examples.peg_insertion.train_rlt \
  --eval_only --eval_episodes 20 --no_residual
```

### What You Get:

Two checkpoints to compare:
1. **VLA-only:** `openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999/`
2. **VLA + RLT:** `checkpoints/rlt_runs/peg_insertion/best.pkl` (SAC agent)

Both are needed for inference — the VLA provides reference actions, the SAC provides corrections.

---

## Phase 3: Collect 50 Demos & Scale

### Step A: Collect 50 Demonstrations

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e

# Full command (equivalent to scripts/collect_50_demos.sh):
cd openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

python scripts/record.py \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.root=/home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets \
    --dataset.num_episodes=50 \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=False \
    --dataset.video=True
```

**Recording Controls:**
| Key | Action |
|-----|--------|
| SPACE | Start recording this episode |
| → (Right) | End & SAVE episode |
| ← (Left) | DISCARD episode |
| ESC | Stop all recording |
| G | Toggle gripper |
| W/S/A/D | Move XY |
| Q/E | Move Z up/down |
| I/K/J/L/U/O | Rotate |

**Workflow per episode:**
1. Position robot at start (SPACE not pressed yet = not recording)
2. Press SPACE → recording starts
3. Perform task (pick peg, insert into hole)
4. Press → → episode saved
5. Repeat

**To resume (add to existing dataset):**
```bash
python scripts/record.py ... --resume
```

### Step B: Symlink Dataset

```bash
mkdir -p ~/.cache/huggingface/lerobot/saifi/
ln -sf /home/robolab-2/ur5e_hande_workspace/openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
    ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual
```

### Step C: Compute Normalization Stats

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
source .venv/bin/activate
.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_fast_ur5e_peg_insertion_lora
```

### Step D: Re-Fine-tune π0-FAST (local, ~5h)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
source .venv/bin/activate
./scripts/train_local.sh pi0_fast_ur5e_peg_insertion_lora --exp-name=peg_insertion_50demos --overwrite
```

### Step E: Re-extract Embeddings & Re-train RL Token

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Extract embeddings from new checkpoint (takes ~2 min)
python -m rlt.models.extract_embeddings \
  --checkpoint openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/29999 \
  --config pi0_fast_ur5e_peg_insertion_lora \
  --dataset saifi/ur5e-peg-insertion-dual \
  --output checkpoints/rl_token/embeddings_peg_insertion_50demos.pt

# Train RL Token model (~10 min)
python -m rlt.models.train_rl_token \
  --embeddings checkpoints/rl_token/embeddings_peg_insertion_50demos.pt \
  --output checkpoints/rl_token/peg_insertion_50demos_v1.pt \
  --token_dim 512 --epochs 2000
```

### Step F: Run Online RL with 50-demo VLA

Update config or pass args:
```bash
python -m rlt.examples.peg_insertion.train_rlt \
  --rl_token_ckpt checkpoints/rl_token/peg_insertion_50demos_v1.pt
```

---

## Phase 4: HPC Multi-VLA Comparison

See `HPC_TRAINING.md` for full details.

```bash
# On HPC:
sbatch scripts/train_hpc_pi0.sh     # π0 LoRA (V100 32GB)
sbatch scripts/train_hpc_pi05.sh    # π0.5 LoRA (V100 32GB)
```

After training, extract embeddings per VLA variant and train RL Tokens.

---

## Key Technical Details

### Action Flow (per chunk)

```
VLA outputs: ã_ref ∈ R^{10×6}  (10 steps × 6D tcp_delta)
SAC outputs: residual ∈ R^{10×6} (normalized to [-1,1])

final_action[i] = ã_ref[i] + residual[i] * [3mm, 3mm, 3mm, 1.1°, 1.1°, 1.1°]
                                              ↑ max correction per step
```

The 7th dimension (gripper) is handled by `GripperCloseEnv` — always closed for peg insertion.

### Observation Space

```
SAC obs = [z_rl(512) | proprio(19) | ã_ref_flat(60)] = 591 dimensions

where:
  z_rl:     RL Token output (compressed VLA embeddings)
  proprio:  tcp_pose(6) + tcp_vel(6) + force(3) + torque(3) + gripper(1)
  ã_ref:    VLA reference chunk flattened (10 steps × 6D)
```

### File Locations

| What | Path |
|------|------|
| Project root | `/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/` |
| VLA checkpoint | `openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999/` |
| RL Token model | `checkpoints/rl_token/peg_insertion_real_v1.pt` |
| SAC checkpoints | `checkpoints/rlt_runs/peg_insertion/best.pkl` |
| Demo dataset | `openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual/` |
| Training config | `rlt/examples/peg_insertion/config.py` |
| SAC agent | `rlt/agents/sac_agent.py` |
| Training loop | `rlt/examples/peg_insertion/train_rlt.py` |

### Python Environments

| Venv | Used For |
|------|----------|
| `ur5e_hil_serl/.venv/` (Python 3.10) | RLT training, SAC, RL Token, robot control |
| `openpi_ur5e/openpi-ur5e/.venv/` (Python 3.11) | VLA server (serve_policy.py) |
| `openpi_ur5e/lerobot_ur5e_gello/.venv/` (Python 3.11) | Demo collection, inference |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `openpi_server: command not found` | Use `.venv/bin/python scripts/serve_policy.py` instead |
| `BLAS support` error (JAX) | Set `export JAX_PLATFORMS=cpu` (SAC is tiny, CPU is fine) |
| VRAM OOM | Kill zombie python processes: `nvidia-smi` then `kill -9 <PID>` |
| VLA server hangs | Check port 8000 not in use: `lsof -i :8000` |
| Dataset symlink error | `ln -sf /full/path ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-dual` |
| `--resume` error | Dataset format changed — start fresh without `--resume` |
| `STOPPED_INNER_OBJECT` | Gripper has object during calibration. Remove peg, clear protective stop on pendant, ensure Remote Control mode |
| `Calibration failed` | Open gripper manually on teach pendant → remove object → retry |
| Broken venv (No module 'lerobot') | Run: `cd openpi_ur5e/lerobot_ur5e_gello && sed -i 's|snap/code/231/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu|.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu|' .venv/pyvenv.cfg .venv/bin/activate` |
