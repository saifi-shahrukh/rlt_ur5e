# RLT (Residual Learning Transfer) — Complete Analysis & Actionable Next Steps

## 📋 Executive Summary

You have **two fully working repositories** and a **detailed implementation guide** (`RLT_UR5e_SETUP_README.md`). The guide describes how to combine:

1. **`openpi_ur5e`** — π0/π0.5 VLA fine-tuning & inference (already trained on peg insertion with 4 episodes, checkpoint exists at `pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999`)
2. **`ur5e_hil_serl`** — HIL-SERL RL training pipeline (already has 194+608+937 demo transitions, trained reward classifier, full impedance controller working)

Into a new **`rlt/`** package that implements the RLT paper's approach: use π0.5 as a base policy and train a small residual RL agent on top.

---

## 🔍 What You Already Have (Verified Working)

### ur5e_hil_serl (RL Side)
| Component | Status | Details |
|-----------|--------|--------|
| UR5e Impedance Controller | ✅ Working | 100Hz forceMode, RTDE, MRP safety clipping |
| Gymnasium Environment | ✅ Working | `UR5eEnv` + `PegInsertionEnv` |
| Cameras (RS D435 + Kinect v2) | ✅ Working | 128×128 resize, serial: 034422070605 / 000631452147 |
| Keyboard Intervention | ✅ Working | `FakeSpaceMouseExpert` via pynput |
| SAC/RLPD Agent (JAX/Flax) | ✅ Working | ResNet-10 encoder, ensemble 2-10 critics, UTD adjustable |
| Wrapper Stack | ✅ Working | GripperClose→Keyboard→RelativeFrame→Quat2Euler→SERLObs→Chunking |
| Demo Buffer | ✅ 1739 transitions | 3 pkl files for peg insertion |
| Reward Classifier | ✅ Trained | 200 success + 2660 failure images, checkpoint at step 150 |
| Agentlace (ZMQ) | ✅ Working | Ports 5588/5589 for actor↔learner |
| Training Script | ✅ Working | `train_rlpd.py` with --learner / --actor flags |

### openpi_ur5e (VLA Side)
| Component | Status | Details |
|-----------|--------|--------|
| π0-FAST Checkpoint | ✅ Trained | 30k steps, peg insertion, at `checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999` |
| π0 LoRA Checkpoint | ⚠️ Started | Only `wandb_id.txt` — training didn't save steps |
| Dataset | ✅ 4 episodes | `saifi/ur5e-peg-insertion-dual`, 1126 frames, 30fps, 7D actions |
| LeRobot Recording | ✅ Working | Keyboard teleop, dual cam (wrist_cam + overview_cam) |
| OpenPI Inference Server | ✅ Working | `scripts/serve_policy.py` with WebSocket client |
| Training Config (pi05) | ✅ Defined | `pi05_ur5e_peg_insertion_lora` config exists |
| PyTorch Model | ✅ Available | `PI0Pytorch` with `PaliGemmaWithExpertModel` |

---

## 🏗️ RLT Architecture (From the README)

The RLT approach bridges these two systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RLT CONTROL LOOP                             │
│                                                                  │
│  1. obs = UR5eEnv.get_obs()  (images + proprio)                 │
│                                                                  │
│  2. z_tokens, ã = Pi05Hook.get_embeddings_and_actions(obs)      │
│     └── Forward hook on paligemma.model.norm → (N_prefix, 2048) │
│     └── Action chunk ã: (50, 7) from diffusion denoising        │
│                                                                  │
│  3. z_rl = RLTokenModel.encode(z_tokens) → (512,)              │
│     └── Transformer encoder with <rl> query token               │
│                                                                  │
│  4. residual_chunk = RLPDAgent.sample([z_rl, proprio, ã[:C]])   │
│     └── JAX SAC actor: MLP(512+7+70) → (70,) = 10×7            │
│     └── BC regularizer: L = -Q + β||a - ã||²                   │
│                                                                  │
│  5. final_action = ã[:C] + residual_chunk  (small residual!)    │
│     └── Execute C=10 steps open-loop on robot                   │
│                                                                  │
│  6. reward = classifier(next_obs) or human label                │
│  7. Store (z_rl, proprio, action, ref, reward) → buffer         │
│  8. Learner: G=5 gradient updates (RLPD 50/50 sampling)        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The RL agent does NOT see raw images. It sees `z_rl` (512-d compressed token from π0.5's internal representations). This makes RL extremely sample-efficient.

---

## 🎯 NEXT STEPS — Prioritized Action Plan

### Phase 0: Foundation (Do This FIRST — No Hardware Needed)
**Time: 1-2 days | Goal: Create the `rlt/` package skeleton and verify imports**

#### Step 0.1: Create the `rlt/` Package Structure
```bash
cd ~/ur5e_hande_workspace/rlt_ur5e

mkdir -p rlt/{models,training,agents,envs,examples/ethernet_insertion,scripts,tests}
touch rlt/__init__.py rlt/models/__init__.py rlt/training/__init__.py \
      rlt/agents/__init__.py rlt/envs/__init__.py
```

#### Step 0.2: Create `pyproject.toml` for the RLT Package
```toml
[project]
name = "rlt"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "numpy",
    "gymnasium",
    "scipy",
]

[tool.setuptools.packages.find]
where = ["."]
```

#### Step 0.3: Implement Core Files (Copy From README)
The README provides **complete, ready-to-use code** for these files:

| File | What it does | Lines in README |
|------|-------------|----------------|
| `rlt/models/pi05_hook.py` | PyTorch hook to extract VLM embeddings during forward pass | ~80 lines |
| `rlt/models/rl_token.py` | Transformer encoder-decoder for compressing embeddings | ~130 lines |
| `rlt/training/train_rl_token.py` | Offline training script for RL token model | ~120 lines |
| `rlt/agents/rlt_sac_agent.py` | Modified RLPD agent with BC regularizer + action chunks | ~150 lines |
| `rlt/agents/rlt_buffer.py` | Chunked replay buffer with stride-2 subsampling | ~80 lines |
| `rlt/envs/ur5e_rlt_env.py` | Env wrapper: VLA inference + RL token extraction | ~120 lines |
| `rlt/examples/ethernet_insertion/config.py` | All hyperparameters | ~60 lines |
| `rlt/examples/ethernet_insertion/actor.py` | Actor node (robot machine) | ~150 lines |
| `rlt/examples/ethernet_insertion/learner.py` | Learner node (GPU machine) | ~130 lines |

#### Step 0.4: Verify Python Environment Compatibility
```bash
cd ~/ur5e_hande_workspace/rlt_ur5e/ur5e_hil_serl
source .venv/bin/activate

# Test existing JAX setup
python -c "import jax; print('JAX devices:', jax.devices())"

# Add PyTorch (for RL token model)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify both work together
python -c "
import jax; import torch
print('JAX:', jax.devices())
print('PyTorch CUDA:', torch.cuda.is_available())
print('Both frameworks loaded successfully!')
"

# Install rlt package
cd ~/ur5e_hande_workspace/rlt_ur5e
pip install -e .
```

#### Step 0.5: Unit Test the RL Token Model (No GPU needed for small test)
```bash
python -c "
import torch
from rlt.models.rl_token import RLTokenModel

model = RLTokenModel(embed_dim=2048, token_dim=512)
x = torch.randn(2, 527, 2048)  # Batch=2, N_prefix=527, embed_dim=2048
loss, z_rl = model.compute_loss(x)
print(f'Loss: {loss.item():.4f}')
print(f'z_rl shape: {z_rl.shape}')  # Should be (2, 512)
print('RL Token model works!')
"
```

---

### Phase 1: Data Collection (Requires Hardware)
**Time: 1-2 days | Goal: Collect enough demos for π0.5 fine-tuning**

#### Current Data Gap
- You have **4 episodes** for peg insertion in LeRobot format
- The RLT README says you need **100 episodes** for VLA fine-tuning
- For the **ethernet insertion** task (the README's target), you have **0 episodes**

#### Decision Point: Task Selection

**Option A: Use Peg Insertion (faster — reuse existing data)**
- Already have 4 LeRobot demos + 1739 SERL transitions + reward classifier
- Need ~50-96 more LeRobot demos (total 100)
- π0-FAST checkpoint already trained (can use as starting point)
- Recommendation: **Start here for validation**

**Option B: Switch to Ethernet Insertion (as in README)**
- Need 100 new episodes from scratch
- Need new reward classifier
- Need new SERL demos
- Recommendation: **Do this after validating on peg insertion**

#### Step 1.1: Collect More Peg Insertion Demos (LeRobot Format)
```bash
cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

# Record ~50 more episodes (start with peg above hole)
python scripts/record.py \
  --robot.type=ur5e_dual_cam \
  --robot.ip=172.22.1.139 \
  --fps=30 \
  --repo-id=saifi/ur5e-peg-insertion-dual \
  --num-episodes=50 \
  --push-to-hub=0 \
  --single-task="Pick up the peg and insert it into the hole."
```

#### Step 1.2: Verify Dataset Size
```bash
python -c "
import json
info = json.load(open('datasets/saifi/ur5e-peg-insertion-dual/meta/info.json'))
print(f'Episodes: {info[\"total_episodes\"]}')
print(f'Frames: {info[\"total_frames\"]}')
assert info['total_episodes'] >= 50, 'Need more demos!'
"
```

---

### Phase 2: π0.5 Fine-Tuning (Requires GPU — HPC or A100)
**Time: 3-12 hours training | Goal: Working base VLA policy**

#### Step 2.1: Train π0.5 on Peg Insertion
```bash
cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
source .venv/bin/activate

# Compute normalization stats for new data
LEROBOT_HOME=~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/datasets \
uv run scripts/compute_norm_stats.py --config-name pi05_ur5e_peg_insertion_lora

# Train (on HPC with V100 32GB+)
LEROBOT_HOME=~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/datasets \
uv run scripts/train.py \
  --config-name pi05_ur5e_peg_insertion_lora \
  --exp-name rlt_base_v1 \
  --overwrite
```

#### Alternative: Use Existing π0-FAST Checkpoint as Base
If you can't train π0.5, the existing π0-FAST checkpoint can work:
```
checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999
```
**Note**: The RLT paper uses π0.5 (with cross-attention and adaRMSNorm), not π0-FAST. But for initial validation, π0-FAST can still produce action chunks that the residual RL can refine.

#### Step 2.2: Evaluate Base Policy Success Rate
```bash
# Serve the checkpoint
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi0_fast_ur5e_peg_insertion_lora \
  --policy.dir=checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999 \
  --port 8000

# In another terminal, run inference on robot
cd ../lerobot_ur5e_gello
python scripts/remote_pi_inference_dual_cam.py \
  --ip=localhost --port=8000 \
  --prompt="Pick up the peg and insert it into the hole." \
  --robot.type=ur5e_dual_cam --robot.ip=172.22.1.139 --fps=30

# Run 20 episodes, record success rate
# Expected: 20-50% (motivates why RLT is needed)
```

---

### Phase 3: RL Token Training (Offline, GPU needed)
**Time: 30-60 minutes | Goal: Trained encoder that compresses VLM tokens**

#### Step 3.1: Pre-compute VLM Embeddings from Demo Data
This runs the π0.5 model on each demo frame and caches the (N_prefix, 2048) embeddings.

```bash
cd ~/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/openpi_ur5e/openpi-ur5e/src:$PYTHONPATH"

python rlt/training/train_rl_token.py \
  --demo_root openpi_ur5e/datasets/saifi/ur5e-peg-insertion-dual \
  --vla_ckpt openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999 \
  --vla_config pi0_fast_ur5e_peg_insertion_lora \
  --save_path checkpoints/rl_token/peg_insertion_v1.pt \
  --steps 5000 \
  --batch_size 32 \
  --token_dim 512
```

#### Step 3.2: Verify RL Token Quality
```bash
python -c "
import torch
ckpt = torch.load('checkpoints/rl_token/peg_insertion_v1.pt', weights_only=False)
print(f'Final loss: {ckpt[\"loss\"]:.5f}')
print(f'Config: {ckpt[\"config\"]}')
assert ckpt['loss'] < 0.05, 'Loss too high — encoder not converged'
print('RL Token model looks good!')
"
```

---

### Phase 4: RLT Online Training (Requires Hardware + GPU)
**Time: 3-5 days | Goal: RL residual that boosts success rate to 90%+**

#### Step 4.1: Prepare the RLT Config for Peg Insertion
Modify `rlt/examples/ethernet_insertion/config.py` (or create `peg_insertion/config.py`):

```python
@dataclass
class Config:
    task = "peg_insertion"
    language = "Pick up the peg and insert it into the hole."
    robot_ip = "172.22.1.139"
    control_hz = 10
    
    # Checkpoints
    vla_ckpt = "openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999"
    vla_config = "pi0_fast_ur5e_peg_insertion_lora"
    rl_token_ckpt = "checkpoints/rl_token/peg_insertion_v1.pt"
    
    # Same env config as your working peg_insertion
    # (copy from ur5e_hil_serl/examples/experiments/peg_insertion/config.py)
    token_dim = 512
    proprio_dim = 19  # tcp_pose(7) + tcp_vel(6) + force(3) + torque(3)
    action_dim = 6    # GripperCloseEnv removes gripper dim
    chunk_size = 10
    
    # Training
    beta = 1.0        # BC regularizer weight
    utd = 5           # Update-to-data ratio
    batch_size = 256
    warmup_episodes = 20
    total_episodes = 800
```

#### Step 4.2: Launch Training (3 Terminals)

```bash
# Terminal 1: OpenPI VLA Server (GPU)
cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
source .venv/bin/activate
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi0_fast_ur5e_peg_insertion_lora \
  --policy.dir=checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_run1/29999 \
  --port 8000

# Terminal 2: RLT Learner (GPU)
cd ~/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PYTHONPATH"
python rlt/examples/peg_insertion/learner.py

# Terminal 3: RLT Actor (Robot machine, can be CPU-only for JAX)
cd ~/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # Needs GPU for VLA hook + RL token
python rlt/examples/peg_insertion/actor.py
```

#### Step 4.3: Monitor Training
```
Expected Learning Curve:
  Ep 0-20:    WARMUP — VLA only, buffer fills.        Baseline SR: 20-40%
  Ep 20-100:  RL starts, critic learning.             SR: 20-50%
  Ep 100-300: Residual deviates from VLA baseline.    SR: 40-65%
  Ep 300-600: Convergence.                            SR: 65-85%
  Ep 600-800: Fine-tuned residual.                    SR: 80-95%
```

---

## ⚠️ Critical Implementation Decisions

### Decision 1: π0-FAST vs π0.5 for Hook
The README assumes **π0.5** (PyTorch, `PI0Pytorch` class). But you have a trained **π0-FAST** checkpoint.

**Issue**: π0-FAST uses a different architecture (FAST tokenizer, discrete action tokens). The hook location may differ.

**Resolution Options**:
- (A) Train π0.5 on HPC (best quality, matches paper exactly)
- (B) Adapt the hook for π0-FAST (the `paligemma.model.norm` layer still exists — SigLIP+Gemma backbone is shared)
- (C) Use π0 (non-FAST) LoRA — you have the config but no saved checkpoint

**Recommendation**: Start with **(B)** — the PaliGemma backbone is the same in both π0-FAST and π0.5. The hook on `paligemma.model.language_model.model.norm` should still produce meaningful VLM embeddings. Then upgrade to π0.5 for final results.

### Decision 2: Action Dimensions
Your existing SERL setup uses **6D actions** (GripperCloseEnv removes gripper dim). The VLA outputs **7D** (includes gripper). The RL token paper uses **7D × chunk_size**.

**Resolution**: Keep gripper in VLA reference chunk (7D), but the RL residual only outputs 6D (position + rotation). Set `gripper_residual = 0` always.

### Decision 3: Observation Format Mismatch
SERL observation: `{"wrist_1": (1,128,128,3), "overview": (1,128,128,3), "state": (1,19)}`
OpenPI expects: `{"observation.images.wrist_cam": (480,640,3), "observation.images.overview_cam": (480,640,3), "observation.state": (7,)}`

**Resolution**: The `Pi05Hook._prep_obs()` method handles this conversion (resize to 224×224 for SigLIP, format state correctly).

### Decision 4: Same Machine or Distributed?
The README shows actor + learner on different machines (agentlace ZMQ). For RLT, you also need the VLA server.

**Simplest setup** (single GPU machine):
- VLA server: runs as a subprocess (port 8000)
- RL Token extraction: runs inline in the actor (same GPU)
- Learner: runs in a separate process (same GPU, JAX)
- Actor: runs inline with the robot (JAX on CPU mode, PyTorch on GPU for VLA)

---

## 🔧 Files to Create (In Priority Order)

| # | File | Purpose | Difficulty |
|---|------|---------|------------|
| 1 | `rlt/models/rl_token.py` | Core RL token encoder-decoder | Easy (copy from README) |
| 2 | `rlt/models/pi05_hook.py` | VLA forward hook for embedding extraction | Medium (adapt for π0-FAST) |
| 3 | `rlt/training/train_rl_token.py` | Offline training script | Easy (copy from README) |
| 4 | `rlt/agents/rlt_buffer.py` | Chunked replay buffer | Easy (copy from README) |
| 5 | `rlt/envs/ur5e_rlt_env.py` | RLT environment wrapper | Medium (bridges SERL+OpenPI) |
| 6 | `rlt/agents/rlt_sac_agent.py` | Modified SAC with BC term | Hard (modify existing JAX agent) |
| 7 | `rlt/examples/peg_insertion/config.py` | All hyperparameters | Easy |
| 8 | `rlt/examples/peg_insertion/actor.py` | Actor training loop | Medium |
| 9 | `rlt/examples/peg_insertion/learner.py` | Learner training loop | Medium |

---

## 🧪 Validation Milestones

| Milestone | Test | Expected Result |
|-----------|------|----------------|
| M1: RL Token trains | `train_rl_token.py` converges | Loss < 0.05 in 5k steps |
| M2: VLA hook works | Run Pi05Hook on 1 image | z_tokens shape (N, 2048), N≈527 |
| M3: End-to-end obs | UR5eRLTEnv.reset() returns valid obs | z_rl(512,) + proprio(19,) + ref(10,7) |
| M4: Residual is small | After 100 RL episodes, ||residual|| | < 5mm per step |
| M5: Improvement | RLT success rate vs VLA-only | RLT > VLA by ≥20% |
| M6: Final | 50-episode evaluation | > 85% success rate |

---

## 📊 Key Numbers to Remember

| Parameter | Value | Source |
|-----------|-------|--------|
| VLA action horizon (H) | 50 (π0.5) or 30 (your π0-FAST config) | `pi0_config.action_horizon` |
| RL chunk size (C) | 10 | RLT paper |
| VLM embed dim | 2048 | Gemma-2B hidden size |
| RL token dim | 512 | Compressed representation |
| N_prefix tokens | ~527 (2 cams × 256 patches + ~15 lang tokens) | SigLIP output |
| Control frequency | 10 Hz | Your ur5e_hil_serl |
| SERL proprio dim | 19 | tcp_pose(7)+vel(6)+force(3)+torque(3) |
| Action dim | 6 (with GripperCloseEnv) or 7 | Config-dependent |
| UTD ratio | 5 (paper) | Lower than RLPD's 20 due to chunk correlation |
| BC weight β | 1.0 (start) | Tune: decrease if too conservative |
| Ref dropout | 50% | Paper: prevents mode collapse to VLA |
| Demo buffer ratio | 50% demo / 50% online | RLPD symmetric sampling |

---

## 🚨 Safety Checklist (Before Real Robot)

- [ ] Test in `fake_env=True` mode first (no hardware)
- [ ] Verify ACTION_SCALE matches: `[0.005, 0.03, 1.0]` for peg insertion
- [ ] Safety box: `ABS_POSE_LIMIT_LOW/HIGH` from your working config
- [ ] First RLT episodes at reduced residual scale (e.g., 0.5× normal)
- [ ] E-stop tested and within reach
- [ ] Impedance controller damping verified (0.1)
- [ ] Force threshold check: auto-truncate at 20N downward
- [ ] VLA server response time < 200ms (otherwise control loop breaks)

---

## 📅 Recommended Timeline

| Day | Phase | Deliverable |
|-----|-------|-------------|
| Day 1 | Phase 0 | `rlt/` package created, imports verified, RL token unit test passes |
| Day 2-3 | Phase 1 | 50+ LeRobot demos collected for peg insertion |
| Day 3-4 | Phase 2 | π0.5 trained on HPC OR π0-FAST validated as base |
| Day 4 | Phase 3 | RL Token model trained, loss < 0.05 |
| Day 5-6 | Phase 4 | RLT actor+learner running, first 100 episodes |
| Day 7-10 | Phase 4 cont. | 800 episodes, success rate converging |
| Day 11-12 | Evaluation | 50-episode eval, ablation (VLA-only vs RLT) |

---

## 🎯 Immediate First Action

**Right now, run this command to create the skeleton:**

```bash
cd ~/ur5e_hande_workspace/rlt_ur5e
mkdir -p rlt/{models,training,agents,envs,examples/peg_insertion,scripts,tests}
touch rlt/__init__.py rlt/models/__init__.py rlt/training/__init__.py \
      rlt/agents/__init__.py rlt/envs/__init__.py
```

Then copy `rlt/models/rl_token.py` from the README (Section Stage B, ~130 lines of code) and run the unit test. That's your first green checkmark. ✅
