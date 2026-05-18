# Model Architectures — π0, π0.5, π0-FAST, and RL Token

This document explains the internal design of each model we train and deploy
for the UR5e peg insertion task.

---

## Table of Contents

1. [Overview](#overview)
2. [π0 (Pi-Zero)](#π0-pi-zero)
3. [π0.5 (Pi-Zero-Point-Five)](#π05-pi-zero-point-five)
4. [π0-FAST (Pi-Zero-FAST)](#π0-fast-pi-zero-fast)
5. [RL Token Model](#rl-token-model)
6. [Comparison Table](#comparison-table)
7. [Training Hyperparameters](#training-hyperparameters)

---

## Overview

All three VLA (Vision-Language-Action) models share the same high-level structure:

    Camera Images (224×224) → SigLIP Vision Encoder → Image Tokens
    Language Prompt → Gemma Tokenizer → Language Tokens
    [Image Tokens + Language Tokens] = "Prefix" (bidirectional attention)
    [State + Noisy Actions + Timestep] = "Suffix" (causal attention)
    Combined → Gemma LLM Backbone → Predicted Velocity/Tokens → Actions

The key differences are:
- π0: Flow matching with separate action expert (Gemma-300M)
- π0.5: Same as π0 but uses AdaRMSNorm for timestep injection
- π0-FAST: Autoregressive token prediction (no flow matching)

---

## π0 (Pi-Zero)

### Architecture

    ┌─────────────────────────────────────────────────────────────┐
    │                        PREFIX                                │
    │  [Image_1 tokens][Image_2 tokens][Language tokens]           │
    │  (256 per image)  (256 per image) (up to 48 tokens)          │
    │  ← Bidirectional self-attention (all attend to each other) → │
    └──────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────┐
    │                        SUFFIX                                │
    │  [State_token][Action_1]...[Action_30]                       │
    │  ← Causal attention (can attend to prefix, not vice versa) → │
    └──────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    action_out_proj → v_t (velocity)

### Components

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| SigLIP (Vision) | ViT-So400m/14, 224×224 input | ~400M (frozen) |
| PaliGemma (VLM) | Gemma-2B transformer | ~2B (LoRA rank=16) |
| Action Expert | Gemma-300M transformer | ~300M (LoRA rank=32) |
| state_proj | Linear(7 → 1024) | 8K |
| action_in_proj | Linear(7 → 1024) | 8K |
| action_time_mlp_in | Linear(2048 → 1024) | 2M |
| action_time_mlp_out | Linear(1024 → 1024) | 1M |
| action_out_proj | Linear(1024 → 7) | 7K |

### How It Works (Flow Matching)

**Training:**
1. Sample noise               with same shape as actions             
2. Sample timestep                                    (biased toward t=1)
3. Interpolate:                                   (noisy actions)
4. Target velocity:                     (direction from clean to noise)
5. Forward pass through model → predicted velocity      
6. Loss =                 averaged over action horizon

**Inference (Denoising):**
1. Start with pure noise                
2. For                iterations:
   - Predict velocity                      
   - Step:                             where                    
3. Return       as the predicted actions

### Attention Pattern

- Prefix tokens (images + language): **bidirectional** — all attend to each other
- Suffix tokens (state + actions): **causal** — prefix cannot see suffix
- State token starts a new attention block (ar_mask=True)
- Action tokens: first is causal break, rest share attention

### Key Design: Two-Expert Architecture

π0 uses TWO Gemma models in a single forward pass:
- Expert 0: PaliGemma (2B) — processes images + language (the "brain")
- Expert 1: Action Expert (300M) — processes state + actions (the "motor")

Both share the same attention pattern but have separate parameters.
The LLM processes                                  as two segments.

---

## π0.5 (Pi-Zero-Point-Five)

### Differences from π0

π0.5 is structurally identical to π0 with two key modifications:

1. **No state token in suffix** — state is encoded as discrete language tokens
   in the prefix (via the                        flag)

2. **AdaRMSNorm for timestep** — instead of concatenating time with actions
   through an MLP, π0.5 injects the timestep via Adaptive RMSNorm:

        # π0: concatenate and MLP
        action_time = MLP([action_tokens || time_tokens])  
        
        # π0.5: AdaRMSNorm conditioning
        time_emb = MLP(sincos_embed(t))      # (B, 1024)
        action_tokens = adaRMSNorm(action_tokens, cond=time_emb)

### Architecture

    ┌─────────────────────────────────────────────────────────────┐
    │                        PREFIX                                │
    │  [Image_1][Image_2][State_as_text][Language_prompt]           │
    │  ← Bidirectional attention →                                 │
    └──────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────┐
    │                        SUFFIX                                │
    │  [Action_1]...[Action_30]  + AdaRMSNorm(time_emb)            │
    │  ← Causal attention →                                        │
    └──────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    action_out_proj → v_t (velocity)

### Components

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| SigLIP (Vision) | ViT-So400m/14, 224×224 input | ~400M (frozen) |
| PaliGemma (VLM) | Gemma-2B transformer | ~2B (LoRA rank=16) |
| Action Expert | Gemma-300M transformer + AdaRMSNorm | ~300M (LoRA rank=32) |
| time_mlp_in | Linear(1024 → 1024) | 1M |
| time_mlp_out | Linear(1024 → 1024) | 1M |
| action_in_proj | Linear(7 → 1024) | 8K |
| action_out_proj | Linear(1024 → 7) | 7K |

### Loss Function

Identical to π0:                 — flow matching velocity loss.

### Why AdaRMSNorm?

AdaRMSNorm allows the timestep to modulate the layer normalization gain,
giving the network a more expressive way to condition on time without
increasing the sequence length. This technique comes from diffusion models
(DiT architecture) and produces slightly better action quality.

---

## π0-FAST (Pi-Zero-FAST)

### Fundamental Difference

π0-FAST is NOT a flow matching model. It's an **autoregressive token predictor**:
- Actions are discretized into tokens using a learned tokenizer
- The model predicts the next action token (like GPT predicts next word)
- No noise, no timestep, no denoising loop

### Architecture

    ┌─────────────────────────────────────────────────────────────┐
    │                      INPUT SEQUENCE                           │
    │  [Image_1 tokens][Image_2 tokens][Prompt + Action tokens]    │
    │  (256 per image)  (256 per image) (up to 180 tokens)         │
    │                                                              │
    │  Attention: images=bidirectional, text+actions=causal         │
    └──────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    Logits → softmax → next token prediction

### Components

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| SigLIP (Vision) | ViT-So400m/14, 224×224 input | ~400M (frozen) |
| PaliGemma (VLM) | Gemma-2B transformer | ~2B (LoRA rank=16) |
| Tokenizer | FAST tokenizer (learned VQ) | External |

Note: NO separate action expert. Only the PaliGemma backbone.

### How It Works (Autoregressive)

**Training:**
1. Tokenize ground-truth actions into discrete tokens using FAST tokenizer
2. Construct input:                                                 
3. Set AR mask: images=bidirectional, prompt+actions=causal
4. Predict next token at each position (shifted by 1)
5. Loss = Cross-entropy on action token positions only (masked)

    loss = -sum(log_softmax(logits) * one_hot(targets) * loss_mask) / sum(loss_mask)

**Inference (Autoregressive Decoding):**
1. Encode prefix (images + prompt) → fill KV cache
2. Generate tokens one-by-one:
   - Sample/argmax from logits
   - Feed back as next input
   - Stop at EOS token or max_decoding_steps
3. Detokenize output tokens → continuous actions

### Loss Function

**Cross-Entropy Loss** (not MSE!):

    targets = one_hot(action_tokens[:, 1:], vocab_size)
    logp = log_softmax(model_logits)
    loss = -sum(targets * logp * loss_mask) / sum(loss_mask)

Only tokens marked by                   contribute to the loss
(image and prompt tokens are excluded).

### Key Differences from π0/π0.5

| Aspect | π0/π0.5 | π0-FAST |
|--------|---------|--------|
| Action representation | Continuous (7-dim) | Discrete tokens |
| Generation | Flow matching (10 steps) | Autoregressive (variable) |
| Loss | MSE (velocity) | Cross-entropy (tokens) |
| Inference speed | ~200-300ms (10 denoise steps) | ~50ms (single pass) |
| Action expert | Gemma-300M (separate) | None (single backbone) |
| Timestep conditioning | Yes (sine-cosine embed) | No |

### Why "FAST"?

FAST = "Fast Action Sequence Tokenizer". The speed advantage:
- No iterative denoising (10 forward passes → 1 forward pass)
- KV cache enables efficient autoregressive decoding
- ~4-6× faster inference than π0/π0.5

---

## RL Token Model

### Purpose

Compress the VLM's internal state (hundreds of high-dim tokens) into a single
compact vector suitable as RL state input. The RL agent (SAC) needs a fixed-size
state vector, not 500+ tokens of dimension 2048.

### Architecture

    VLM Prefix Embeddings (B, N, 2048)
           │
           ▼
    ┌─────────────────────────────────────┐
    │          ENCODER                     │
    │  [token_1, ..., token_N, <rl>]       │  ← learnable query appended
    │  + positional embeddings             │
    │  → 4-layer Transformer (Pre-Norm)    │
    │  → extract output at <rl> position   │
    │  → Linear(2048 → 512)                │  ← information bottleneck
    └──────────────┬──────────────────────┘
                   │
                   ▼
              z_rl (B, 512)  ← THIS is the RL state
                   │
                   ▼ (training only)
    ┌─────────────────────────────────────┐
    │          DECODER                     │
    │  Linear(512 → 2048)                  │  ← expand back
    │  Cross-attention to z_rl             │
    │  Teacher-forced: [BOS, t1,...,tN-1]   │
    │  → 4-layer Transformer Decoder       │
    │  → Linear(2048 → 2048)               │
    │  → z_hat (B, N, 2048)                │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    Loss = MSE(z_hat, stop_gradient(z_input))

### Components

| Component | Details | Parameters |
|-----------|---------|------------|
| rl_query | Learnable (1, 1, 2048) | 2K |
| enc_pos | Embedding(max_len+1, 2048) | ~2M |
| encoder | 4-layer TransformerEncoder, 8 heads, FFN=2048 | ~67M |
| enc_norm | LayerNorm(2048) | 4K |
| to_token | Linear(2048 → 512) | 1M |
| from_token | Linear(512 → 2048) | 1M |
| dec_pos | Embedding(max_len, 2048) | ~2M |
| bos | Learnable (1, 1, 2048) | 2K |
| decoder | 4-layer TransformerDecoder, 8 heads, FFN=2048 | ~101M |
| dec_norm | LayerNorm(2048) | 4K |
| out_head | Linear(2048 → 2048) | 4M |
| **Total** | | **~279M** |

### Loss Function

**MSE Reconstruction Loss with Stop-Gradient:**

    z_sg = stop_gradient(z_input)        # Don't backprop into VLM
    z_rl = encoder(z_input)              # Gradient flows through encoder
    z_hat = decoder(z_rl, z_sg)          # Teacher-forced reconstruction
    loss = MSE(z_hat, z_sg)              # Reconstruct originals

The stop-gradient is critical: it creates an **information bottleneck**.
The encoder must compress ALL relevant information into the 512-dim z_rl
because the decoder can only access z_rl (not the original tokens).

### Inference (Deployment)

At inference time, only the **encoder** is used:

    # During RL rollout:
    vlm_embeddings = policy.embed_prefix(observation)  # (1, N, 2048)
    z_rl = rl_token_model.encode(vlm_embeddings)       # (1, 512)
    action = sac_agent.act(z_rl)                       # SAC policy

The decoder is discarded after training.

### Design Rationale

Why not just use mean-pooling of VLM tokens?
- Mean-pooling loses positional/structural information
- The learned query token can attend selectively to relevant tokens
- The reconstruction objective ensures nothing important is lost
- 512-dim is much more manageable for SAC than 500×2048 = 1M-dim

---

## Comparison Table

| Property | π0 | π0.5 | π0-FAST | RL Token |
|----------|-----|------|---------|----------|
| Framework | JAX/Flax | JAX/Flax | JAX/Flax | PyTorch |
| Total params | ~3B | ~3B | ~2.5B | ~279M |
| Trainable (LoRA) | ~50M | ~50M | ~20M | ~279M (full) |
| Vision encoder | SigLIP-So400m | SigLIP-So400m | SigLIP-So400m | None |
| LLM backbone | Gemma-2B | Gemma-2B | Gemma-2B | N/A |
| Action expert | Gemma-300M | Gemma-300M | None | N/A |
| Action type | Continuous | Continuous | Discrete tokens | N/A |
| Generation | Flow matching | Flow matching | Autoregressive | N/A |
| Loss | MSE (velocity) | MSE (velocity) | Cross-entropy | MSE (recon) |
| Inference speed | ~200ms | ~300ms | ~50ms | ~5ms |
| Images | 2× 224×224 | 2× 224×224 | 2× 224×224 | N/A |
| Action dim | 7 | 7 | 7 | N/A |
| Action horizon | 30 | 30 | 30 | N/A |
| Output | 30×7 actions | 30×7 actions | 30×7 actions | 512-dim z_rl |

---

## Training Hyperparameters

### VLA Models (v2, 4-GPU FSDP)

| Parameter | π0 v2 | π0.5 v2 | π0-FAST v2 |
|-----------|-------|---------|------------|
| Steps | 30,000 | 30,000 | 30,000 |
| Batch size | 8 | 8 | 8 |
| Per-GPU batch | 2 | 2 | 2 |
| Learning rate | Default (cosine) | Default (cosine) | Default (cosine) |
| VLM LoRA rank | 16 | 16 | 16 |
| Expert LoRA rank | 32 | 32 | N/A |
| Image resolution | 224×224 | 224×224 | 224×224 |
| Cameras | 2 (overhead + wrist) | 2 (overhead + wrist) | 2 (overhead + wrist) |
| Dataset | 50 demos | 50 demos | 50 demos |
| Save interval | 500 steps | 500 steps | 500 steps |
| Keep period | 5000 steps | 5000 steps | 5000 steps |
| EMA | None | None | None |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| Optimizer | AdamW (default) | AdamW (default) | AdamW (default) |
| FSDP | 4 GPUs | 4 GPUs | 4 GPUs |

### RL Token Model

| Parameter | Value |
|-----------|-------|
| Steps | 5,000 |
| Batch size | 4 (effective 32 with grad_accum=8) |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Embed dim | 2048 |
| Token dim | 512 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 8 |
| FFN dim | 2048 |
| Dropout | 0.0 |
| Sequence length | Model-specific (816-968) |

---

## How They Work Together

### Training Pipeline

    1. Fine-tune VLA (π0/π0.5/π0-FAST) on demos  →  Learns task behavior
    2. Extract VLM embeddings from VLA            →  Cache prefix tokens
    3. Train RL Token on cached embeddings        →  Learn compression
    4. Deploy VLA + RL Token for online RL        →  SAC uses z_rl as state

### Inference Pipeline (Online RL)

    Camera Images + Prompt
          │
          ▼
    VLA Model (serve_policy.py)
          │
          ├──→ Actions (sent to robot)
          │
          └──→ VLM Prefix Embeddings
                    │
                    ▼
              RL Token Encoder
                    │
                    ▼
              z_rl (512-dim)
                    │
                    ▼
              SAC Agent → Residual Action
                    │
                    ▼
              Final Action = VLA_action + residual
