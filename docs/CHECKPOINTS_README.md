# Checkpoints Directory Structure

This folder contains all trained models from HPC for deployment on the Linux workstation.

## Directory Layout

    checkpoints_from_hpc/
    ├── rl_token/                                    ← RL Token models + cached embeddings
    │   ├── pi0_ur5e_peg_insertion_lora_rl_token.pt        (1.1G) Trained encoder-decoder
    │   ├── pi05_ur5e_peg_insertion_lora_rl_token.pt       (1.1G) Trained encoder-decoder
    │   ├── pi0_fast_ur5e_peg_insertion_lora_rl_token.pt   (1.1G) Trained encoder-decoder
    │   ├── embeddings_pi0_ur5e_..._step4000.pt            (1.3G) Cached VLM embeddings
    │   ├── embeddings_pi05_ur5e_..._step4999.pt           (1.5G) Cached VLM embeddings
    │   └── embeddings_pi0_fast_ur5e_..._step4999.pt       (1.5G) Cached VLM embeddings
    │
    ├── pi0_ur5e_peg_insertion_lora/                  ← VLA checkpoint (pi0)
    │   └── peg_insertion_50demos/4000/
    │       ├── params/                              (5.8G) LoRA fine-tuned weights
    │       └── assets/saifi/.../norm_stats.json     Normalization statistics
    │
    ├── pi05_ur5e_peg_insertion_lora/                 ← VLA checkpoint (pi0.5)
    │   └── peg_insertion_50demos/4999/
    │       ├── params/                              (6.0G) LoRA fine-tuned weights
    │       └── assets/saifi/.../norm_stats.json     Normalization statistics
    │
    └── pi0_fast_ur5e_peg_insertion_lora/             ← VLA checkpoint (pi0-FAST)
        └── peg_insertion_50demos/4999/
            ├── params/                              (5.2G) LoRA fine-tuned weights
            └── assets/saifi/.../norm_stats.json     Normalization statistics


## What Each File Does

### VLA Checkpoints (params/ directories)
These are the fine-tuned Vision-Language-Action models. They generate robot actions
from camera images + language prompts. Used by the OpenPI inference server.

- pi0: Balanced quality/speed (~200ms/action)
- pi0.5: Best quality (~300ms/action)
- pi0-FAST: Fastest inference (~50ms/action)

### RL Token Models (*_rl_token.pt)
These compress VLM prefix embeddings (N x 2048) into a single z_rl vector (512-dim).
Used as the state representation for SAC reinforcement learning.

Architecture:
- Encoder: 4-layer Transformer + learnable query token + linear projection
- Decoder: 4-layer Transformer (for training reconstruction loss)
- Input: VLM prefix embeddings (816-968 tokens x 2048 dim)
- Output: z_rl (1 x 512) compressed state

### Embeddings (embeddings_*.pt)
Pre-extracted VLM prefix embeddings from demo observations. Used to train
RL Token models offline. NOT needed for deployment, but kept for reference
and potential retraining.


## Setup on Linux Workstation (robolab-2)

### 1. Place checkpoints

    # VLA checkpoints go into openpi's checkpoint directory
    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
    ln -sf ~/ur5e_hande_workspace/rlt_ur5e/checkpoints_from_hpc/pi0_ur5e_peg_insertion_lora checkpoints/
    ln -sf ~/ur5e_hande_workspace/rlt_ur5e/checkpoints_from_hpc/pi05_ur5e_peg_insertion_lora checkpoints/
    ln -sf ~/ur5e_hande_workspace/rlt_ur5e/checkpoints_from_hpc/pi0_fast_ur5e_peg_insertion_lora checkpoints/

    # RL Token models go into rlt checkpoint directory
    ln -sf ~/ur5e_hande_workspace/rlt_ur5e/checkpoints_from_hpc/rl_token ~/ur5e_hande_workspace/rlt_ur5e/checkpoints/rl_token

### 2. Serve VLA Policy (choose one)

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
    source .venv/bin/activate

    # pi0-FAST (recommended for real-time control):
    python scripts/serve_policy.py --port 8000 \
      policy:checkpoint \
      --policy.config=pi0_fast_ur5e_peg_insertion_lora \
      --policy.dir=checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

    # pi0.5 (best quality):
    python scripts/serve_policy.py --port 8000 \
      policy:checkpoint \
      --policy.config=pi05_ur5e_peg_insertion_lora \
      --policy.dir=checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

    # pi0 (balanced):
    python scripts/serve_policy.py --port 8000 \
      policy:checkpoint \
      --policy.config=pi0_ur5e_peg_insertion_lora \
      --policy.dir=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000

### 3. Load RL Token for SAC

    import torch
    from rlt.models.rl_token import RLTokenModel

    ckpt = torch.load("checkpoints/rl_token/pi0_fast_ur5e_peg_insertion_lora_rl_token.pt")
    model = RLTokenModel(
        embed_dim=ckpt["config"]["embed_dim"],   # 2048
        token_dim=ckpt["config"]["token_dim"],   # 512
        max_len=ckpt["config"]["max_len"],       # model-specific
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Encode VLM embeddings -> z_rl
    z_rl = model.encode(embeddings)  # (batch, 512)


## Training Details

| Model | Steps | Dataset | LoRA Rank | Batch | Speed |
|-------|-------|---------|-----------|-------|-------|
| pi0 | 4000/30000 | 50 demos | VLM=16, Expert=32 | 8 | ~10s/step |
| pi0.5 | 4999/30000 | 50 demos | VLM=16, Expert=32 | 4+accum2 | ~11s/step |
| pi0-FAST | 4999/30000 | 50 demos | VLM=4 | 8 | ~10s/step |
| RL Token (all) | 5000/5000 | 200 embeddings | N/A | 4 | ~1.1 steps/s |

All models are functional at current steps but will improve with more training.
V2 training (2-camera, proper ranks) is running on HPC.
