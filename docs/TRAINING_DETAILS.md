# Training Details: Peg Insertion (50 Demos)

## LoRA Configuration

### pi0 (Diffusion-based continuous actions)

| Component | Variant | LoRA Rank | LoRA Alpha | Params |
|-----------|---------|-----------|------------|--------|
| PaliGemma VLM (2B) | gemma_2b_lora | 16 | 16.0 | ~2.5B |
| Action Expert (300M) | gemma_300m_lora | 32 | 32.0 | ~311M |
| SigLIP Vision Encoder | frozen | - | - | ~400M |
| **Total** | | | | **3.29B** |
| **Trainable (LoRA)** | | | | **468M (14.2%)** |

LoRA is applied to both attention (Q/K/V/O projections) and FFN (gating + linear).

### pi0-FAST (Discrete action tokens)

| Component | Variant | LoRA Rank | LoRA Alpha | Params |
|-----------|---------|-----------|------------|--------|
| PaliGemma VLM (2B) | gemma_2b_lora | **4** (overridden) | 4.0 | ~2.5B |
| FAST Tokenizer | frozen | - | - | ~400M |
| SigLIP Vision Encoder | frozen | - | - | ~400M |
| **Total** | | | | **2.93B** |
| **Trainable (LoRA)** | | | | **421M (14.4%)** |

Note: The CURRENT peg insertion pi0-FAST uses rank=4 (originally configured for
local 16GB GPU). For FUTURE tasks on HPC, use rank=16 (same as pi0) via the
                           config variant. This ensures proper RLT token decoding
capability and matches the base model's default LoRA structure.

### pi0.5 (Cross-attention architecture)

| Component | Variant | LoRA Rank | LoRA Alpha | Params |
|-----------|---------|-----------|------------|--------|
| PaliGemma VLM (2B) | gemma_2b_lora | 16 | 16.0 | ~2.5B |
| Action Expert (300M) | gemma_300m_lora | 32 | 32.0 | ~311M |
| Cross-attention layers | included in VLM | 16 | 16.0 | - |
| SigLIP Vision Encoder | frozen | - | - | ~400M |
| **Total** | | | | **3.28B** |
| **Trainable (LoRA)** | | | | **468M (14.3%)** |

pi0.5 differs from pi0 by adding cross-attention between VLM and action expert,
and using adaRMSNorm for flow-matching timestep injection.

---

## Training Configuration Used on HPC

| Parameter | pi0 | pi0.5 | pi0-FAST |
|-----------|-----|-------|----------|
| Total steps | 5000 | 5000 | 5000 |
| Batch size | 8 | 4 | 8 |
| Grad accumulation | 1 | 2 | 1 |
| Effective batch | 8 | 8 | 8 |
| Num workers | 4 | 4 | 4 |
| Learning rate | cosine decay | cosine decay | cosine decay |
| Optimizer | AdamW | AdamW | AdamW |
| Weight decay | default | default | default |
| EMA decay | None | None | None |
| Precision | bfloat16 (frozen), float32 (LoRA) | same | same |
| Save interval | 1000 steps | 1000 steps | 1000 steps |
| Keep period | 5000 | 5000 | 5000 |
| Action horizon | 30 | 30 | 30 |
| Action dim | 7 (6 joints + 1 gripper) | 7 | 7 |
| Delta actions | Yes (first 6 dims) | Yes | Yes |

---

## What Happens During Training (Step by Step)

### Forward Pass:
1. **Image encoding**: SigLIP processes 3 camera images (224x224) into 256 patch tokens each
2. **Token embedding**: PaliGemma embeds image tokens + language prompt tokens
3. **State projection**: Robot joint state (7-dim) is projected to model dimension
4. **VLM processing**: Gemma-2B transformer processes all tokens through 18 layers
5. **Action head**:
   - pi0/pi0.5: Diffusion head predicts noise for flow matching
   - pi0-FAST: Autoregressive head generates discrete action tokens

### Loss Computation:
- **pi0/pi0.5**: MSE loss between predicted and target noise (flow matching objective)
- **pi0-FAST**: Cross-entropy loss on discrete action token predictions

### Backward Pass:
- Gradients flow only through LoRA weights (A and B matrices)
- Frozen params (bf16) don't receive gradients
- LoRA params (f32) are updated with AdamW

### Checkpoint Saving:
- Every 1000 steps: saves full state (params + optimizer + step)
- Only keeps latest checkpoint (max_to_keep=1)
- Plus keeps every 5000th step permanently (keep_period=5000)
- Final checkpoint at step 4999 (0-indexed)

---

## Training Curves (What to Expect on W&B)

### Loss:
- Starts high (~2.0-5.0 depending on model)
- Drops rapidly in first 500 steps
- Slow convergence from step 1000-5000
- Final loss typically ~0.1-0.5 for well-converged LoRA

### Grad Norm:
- Should be stable (0.01-1.0 range)
- Spikes indicate instability (rare with LoRA)

### Param Norm:
- Slowly increases as LoRA weights grow from zero initialization
- LoRA B matrix starts at zero, A at random

---

## Actual Timing Results (V100 32GB)

| Model | Rate | Compilation (1st step) | Steady State | Total (5000 steps) |
|-------|------|----------------------|-------------|-------------------|
| pi0 | 9.3s/step | ~40s | ~9.3s | ~13 hr |
| pi0.5 | 10.1s/step | ~45s | ~10.1s | ~14 hr |
| pi0-FAST | 10.6s/step | ~40s | ~10.6s | ~14.7 hr |

Note: XLA compilation on first step is cached. Resume jobs start fast.

---

## Dataset Details

- **Name**: saifi/ur5e-peg-insertion-dual (on HuggingFace)
- **Demos**: 50 episodes
- **Cameras**: 2 (overview_cam + wrist_cam, wrist duplicated to left/right)
- **State**: 7-dim (6 joints + 1 gripper)
- **Actions**: 7-dim delta (6 joint deltas + 1 absolute gripper)
- **Episode length**: ~100-200 steps each
- **Total frames**: ~5000-10000
- **Task prompt**: "Pick up the peg and insert it into the hole."

## W&B Project

- URL: https://wandb.ai/saifi/openpi
- Runs:
  - pi0.5: COMPLETE (run n2793kso, 10h 30m)
  - pi0: resumed from step 4000
  - pi0-FAST: resumed from step 4000
