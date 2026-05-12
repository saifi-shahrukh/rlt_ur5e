# Deployment Guide: VLA + RL Token on Linux Workstation

This guide covers transferring trained checkpoints from HPC to the physical
Linux workstation (robolab-2) and running inference.

## Architecture Overview

### VLA Models (trained on HPC, served for inference)

| Model | Config Name | LoRA Rank | Speed | Best For |
|-------|------------|-----------|-------|----------|
| pi0 | pi0_ur5e_peg_insertion_lora | VLM=16, Expert=32 | ~200ms | Balanced |
| pi0.5 | pi05_ur5e_peg_insertion_lora | VLM=16, Expert=32 | ~300ms | Best quality |
| pi0-FAST | pi0_fast_ur5e_peg_insertion_lora | VLM=4 | ~50ms | Fastest |

### RL Token Model (compresses VLM embeddings for RL)

- Input: VLM prefix embeddings (N_tokens x 2048)
- Output: z_rl (1 x 512) compressed representation
- Architecture: 4-layer Transformer encoder with learnable query token
- Encoder params: ~104M, Decoder params: ~175M, Total: ~279M
- Token dim: 512 (the bottleneck / RL state representation)
- Embed dim: 2048 (matches Gemma-2B hidden size)
- Training: MSE reconstruction loss (autoencoder)

### Model-specific sequence lengths

| Model | N_tokens | max_len | Notes |
|-------|----------|---------|-------|
| pi0 | 816 | 826 | 3 images x 256 + language |
| pi0.5 | 968 | 978 | 4 images x 256 + language |
| pi0-FAST | 948 | 958 | 4 images x 256 + language (slightly different tokenizer) |

## Checkpoint Locations on HPC

    /data/beegfs/home/saifi/rlt_ur5e/
    ├── openpi_ur5e/openpi-ur5e/checkpoints/
    │   ├── pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000/
    │   │   ├── params/          <- LoRA weights (needed for inference)
    │   │   └── assets/          <- norm_stats.json
    │   ├── pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/
    │   │   ├── params/
    │   │   └── assets/
    │   └── pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/
    │       ├── params/
    │       └── assets/
    └── checkpoints/rl_token/
        ├── embeddings_pi0_ur5e_peg_insertion_lora_step4000.pt      (1.3G)
        ├── embeddings_pi05_ur5e_peg_insertion_lora_step4999.pt     (1.5G)
        ├── embeddings_pi0_fast_ur5e_peg_insertion_lora_step4999.pt (1.5G)
        ├── pi0_ur5e_peg_insertion_lora_rl_token.pt                 (~1.1G)
        ├── pi05_ur5e_peg_insertion_lora_rl_token.pt                (~1.1G)
        └── pi0_fast_ur5e_peg_insertion_lora_rl_token.pt            (~1.1G)

## Transfer: HPC -> WSL2 -> Linux Workstation

Transfer topology: HPC <-> WSL2 (r10028) <-> Ubuntu (robolab-2)
Direct HPC->Ubuntu is blocked by network segmentation.

### Step 1: HPC -> WSL2 (run from WSL2)

    mkdir -p ~/hpc_checkpoints/rl_token
    mkdir -p ~/hpc_checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000
    mkdir -p ~/hpc_checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999
    mkdir -p ~/hpc_checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

    # RL Token models (small, ~1.1G each)
    scp saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/checkpoints/rl_token/*_rl_token.pt ~/hpc_checkpoints/rl_token/

    # VLA params (needed for inference server)
    rsync -avz --progress saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000/params/ ~/hpc_checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000/params/
    rsync -avz --progress saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000/assets/ ~/hpc_checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000/assets/

    rsync -avz --progress saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/params/ ~/hpc_checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/params/
    rsync -avz --progress saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/assets/ ~/hpc_checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/assets/

    rsync -avz --progress saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/params/ ~/hpc_checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/params/
    rsync -avz --progress saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/assets/ ~/hpc_checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999/assets/

### Step 2: WSL2 -> Linux Workstation (run from WSL2)

    # VLA checkpoints
    rsync -avz --progress ~/hpc_checkpoints/pi0_ur5e_peg_insertion_lora/ robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_ur5e_peg_insertion_lora/
    rsync -avz --progress ~/hpc_checkpoints/pi05_ur5e_peg_insertion_lora/ robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi05_ur5e_peg_insertion_lora/
    rsync -avz --progress ~/hpc_checkpoints/pi0_fast_ur5e_peg_insertion_lora/ robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/

    # RL Token models
    rsync -avz --progress ~/hpc_checkpoints/rl_token/ robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/checkpoints/rl_token/

### Expected sizes

| File | Size | Needed for |
|------|------|------------|
| params/ (per model) | ~500MB-1.5GB | VLA inference server |
| assets/norm_stats.json | <1KB | Normalization during inference |
| *_rl_token.pt | ~1.1GB | RL Token encoder (SAC state) |
| embeddings_*.pt | 1.3-1.5GB | NOT needed on workstation (only for retraining) |

## Deploy on Linux Workstation (robolab-2)

### 1. Serve VLA Policy

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
    source .venv/bin/activate

    # Option A: pi0-FAST (fastest, ~50ms/action)
    python scripts/serve_policy.py --port 8000 \
      policy:checkpoint \
      --policy.config=pi0_fast_ur5e_peg_insertion_lora \
      --policy.dir=checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

    # Option B: pi0.5 (best quality, ~300ms/action)
    python scripts/serve_policy.py --port 8000 \
      policy:checkpoint \
      --policy.config=pi05_ur5e_peg_insertion_lora \
      --policy.dir=checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

    # Option C: pi0 (balanced, ~200ms/action)
    python scripts/serve_policy.py --port 8000 \
      policy:checkpoint \
      --policy.config=pi0_ur5e_peg_insertion_lora \
      --policy.dir=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000

### 2. Load RL Token for SAC Training

    import torch
    from rlt.models.rl_token import RLTokenModel

    # Load the trained RL Token model
    ckpt = torch.load("checkpoints/rl_token/pi0_fast_ur5e_peg_insertion_lora_rl_token.pt")
    model = RLTokenModel(
        embed_dim=ckpt["config"]["embed_dim"],
        token_dim=ckpt["config"]["token_dim"],
        max_len=ckpt["config"]["max_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Encode VLM embeddings to z_rl
    # embeddings shape: (batch, N_tokens, 2048)
    z_rl = model.encode(embeddings)  # -> (batch, 512)

### 3. Full RL Training Loop (SAC + RL Token)

    # See rlt/examples/peg_insertion/train_rlt.py for the full example
    python -m rlt.examples.peg_insertion.train_rlt \
      --vla_config pi0_fast_ur5e_peg_insertion_lora \
      --vla_port 8000 \
      --rl_token_path checkpoints/rl_token/pi0_fast_ur5e_peg_insertion_lora_rl_token.pt

## Important Notes

### pi0-FAST uses LoRA rank=4 (not 16)

The current pi0-FAST checkpoint (step 4999) was trained with paligemma_lora_rank=4.
The config.py has been updated to match. Do NOT change this for the current checkpoint.
Future retraining should use rank=16 (the default).

### norm_stats.json must be in checkpoint assets/

The inference server looks for norm_stats at:
    checkpoints/{config}/peg_insertion_50demos/{step}/assets/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json

If missing, run on HPC:
    bash hpc/fix_checkpoints_for_inference.sh

Or manually copy from:
    openpi-ur5e/assets/{config_name}/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json
Into:
    checkpoints/{config}/peg_insertion_50demos/{step}/assets/saifi/ur5e-peg-insertion-50demos-v2/norm_stats.json

### Embeddings use random observations (not real demos)

The extraction used random observations (used_real_demos: False) because loading
the LeRobot dataset on HPC was complex. For production, re-extract with real demo
frames for better RL Token quality. This is fine for pipeline validation.

### Training steps reference

| Model | Current step | Target | Status |
|-------|-------------|--------|--------|
| pi0 | 4000 | 30000 | Paused (usable for testing) |
| pi0.5 | 4999 | 30000 | Paused (usable for testing) |
| pi0-FAST | 4999 | 30000 | Paused (usable for testing) |

All models are functional at current steps but will improve with more training.
Resume with: sbatch hpc/slurm/{model}_50demos_resume.sh
