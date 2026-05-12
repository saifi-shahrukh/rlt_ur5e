# Guide: Training a New Task (e.g., USB Pick-and-Insert)

This guide walks through how to train all 3 VLA models (pi0, pi0.5, pi0-FAST)
on a new task using the existing HPC infrastructure.

## Prerequisites

- HPC setup already working (sysroot, DT_RPATH, offline caches)
- HuggingFace account with PaliGemma access
- W&B account for logging
- Robot workstation for data collection

---

## Step 1: Collect Demonstration Data

On the robot workstation, use the LeRobot/GELLO teleoperation setup:

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
    source .venv/bin/activate
    python scripts/record.py \
        --robot-path lerobot_robot_ur5e \
        --teleop-path lerobot_teleoperator_gello \
        --repo-id saifi/ur5e-usb-insertion \
        --num-episodes 50

Recommendations:
- Collect at least 50 demonstrations
- Vary starting positions slightly
- Include both cameras (overview + wrist)
- Set meaningful task description in the prompt

---

## Step 2: Push Dataset to HuggingFace

    huggingface-cli upload saifi/ur5e-usb-insertion ./data/saifi/ur5e-usb-insertion

Or if using LeRobot's built-in push:

    python -m lerobot.scripts.push_dataset_to_hub \
        --repo-id saifi/ur5e-usb-insertion

---

## Step 3: Add Training Configs

Edit: openpi_ur5e/openpi-ur5e/src/openpi/training/config.py

Add 3 new configs (copy from peg insertion, modify dataset name):

    # ─── pi0 USB Insertion ─────────────────────────────────────────
    TrainConfig(
        name="pi0_ur5e_usb_insertion_lora",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",         # rank=16 (default)
            action_expert_variant="gemma_300m_lora",   # rank=32 (default)
            action_horizon=30,
        ),
        data=LeRobotUR5DualCamDataConfig(
            repo_id="saifi/ur5e-usb-insertion",        # <-- YOUR DATASET
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="saifi/ur5e-usb-insertion"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=5_000,
        save_interval=1_000,
        batch_size=8,
        num_workers=4,
    ),

    # ─── pi0-FAST USB Insertion (rank=16 for HPC) ──────────────────
    # IMPORTANT: Use rank=16 (NOT rank=4) for HPC training!
    # rank=4 was only for local 16GB GPU. rank=16 matches the base
    # model and is required for proper RLT token decoding.
    TrainConfig(
        name="pi0_fast_ur5e_usb_insertion_lora",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=30,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
            # paligemma_lora_rank=16,  # Use default (16), do NOT set to 4!
        ),
        data=LeRobotUR5DualCamDataConfig(
            repo_id="saifi/ur5e-usb-insertion",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="saifi/ur5e-usb-insertion"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=30, max_token_len=180,
            paligemma_variant="gemma_2b_lora",
            # paligemma_lora_rank=16,  # matches above
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=5_000,
        save_interval=1_000,
        batch_size=8,
        num_workers=4,
    ),

    # ─── pi0.5 USB Insertion ───────────────────────────────────────
    TrainConfig(
        name="pi05_ur5e_usb_insertion_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
        ),
        data=LeRobotUR5DualCamDataConfig(
            repo_id="saifi/ur5e-usb-insertion",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="saifi/ur5e-usb-insertion"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        keep_period=5_000,
        save_interval=1_000,
        batch_size=4,
        grad_accumulation_steps=2,
        num_workers=4,
    ),

KEY NOTES on LoRA Rank:
- pi0: rank=16 (VLM) + rank=32 (action expert) -- defaults from gemma_2b_lora
- pi0-FAST: rank=16 (VLM) -- do NOT override to rank=4 on HPC!
  - rank=4 was a memory hack for 16GB local GPU
  - rank=16 is needed for proper token decoding and RLT compatibility
  - V100 32GB has plenty of memory for rank=16
- pi0.5: rank=16 (VLM) + rank=32 (action expert) -- same as pi0

---

## Step 4: Compute Normalization Statistics

On HPC headnode (has network access to download dataset):

    cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e
    source .venv/bin/activate

    # For pi0:
    python scripts/compute_norm_stats.py --config pi0_ur5e_usb_insertion_lora

    # For pi0-FAST:
    python scripts/compute_norm_stats.py --config pi0_fast_ur5e_usb_insertion_lora

    # For pi0.5:
    python scripts/compute_norm_stats.py --config pi05_ur5e_usb_insertion_lora

This creates:
    assets/pi0_ur5e_usb_insertion_lora/saifi/ur5e-usb-insertion/norm_stats.json
    assets/pi0_fast_ur5e_usb_insertion_lora/saifi/ur5e-usb-insertion/norm_stats.json
    assets/pi05_ur5e_usb_insertion_lora/saifi/ur5e-usb-insertion/norm_stats.json

Commit these to git!

---

## Step 5: Transfer Dataset to HPC

The dataset needs to be in the HPC cache:

    # On headnode (with internet):
    python -c "
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('saifi/ur5e-usb-insertion')
    print(f'Downloaded: {len(ds)} frames')
    "

Or manually:

    bash hpc/02_transfer_dataset.sh

---

## Step 6: Create SLURM Training Scripts

Copy and modify from peg insertion:

    # Copy templates
    cp hpc/slurm/pi0_50demos.sh hpc/slurm/pi0_usb.sh
    cp hpc/slurm/pi0_fast_50demos.sh hpc/slurm/pi0_fast_usb.sh
    cp hpc/slurm/pi05_50demos.sh hpc/slurm/pi05_usb.sh

In each file, change:
    CONFIG="pi0_ur5e_usb_insertion_lora"     # (or pi0_fast/pi05 variant)
    EXP_NAME="usb_insertion_50demos"

For pi0-FAST, REMOVE the paligemma_lora_rank=4 override (use default 16).

---

## Step 7: Submit Training

    sbatch hpc/slurm/pi0_usb.sh
    sbatch hpc/slurm/pi0_fast_usb.sh
    sbatch hpc/slurm/pi05_usb.sh

Expected times (5000 steps, V100 32GB):
- pi0: ~13 hours (may need resume)
- pi0-FAST: ~14 hours (may need resume)
- pi0.5: ~14 hours (may need resume)

To guarantee completion in 12 hours, use:
    --num-train-steps=3000 --grad-accumulation-steps=2

---

## Step 8: Resume if Needed

If jobs hit the 12-hour limit:

    # Create resume scripts (copy from peg insertion)
    cp hpc/slurm/pi0_50demos_resume.sh hpc/slurm/pi0_usb_resume.sh
    # Edit: change CONFIG and EXP_NAME
    sbatch hpc/slurm/pi0_usb_resume.sh

Key: use --resume flag, NOT --overwrite

---

## Step 9: Transfer to Robot Workstation

Path: HPC -> WSL2 (r10028) -> Physical Ubuntu (172.22.1.188)

    # Step 1: HPC -> WSL2
    scp -r saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_ur5e_usb_insertion_lora ~/hpc_checkpoints/

    # Step 2: WSL2 -> Physical Ubuntu
    scp -r ~/hpc_checkpoints/pi0_ur5e_usb_insertion_lora robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/

---

## Step 10: Deploy and Test

On the physical Ubuntu robot workstation:

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
    source .venv/bin/activate

    python scripts/serve_policy.py --port 8000 \
        policy:checkpoint \
        --policy.config=pi0_fast_ur5e_usb_insertion_lora \
        --policy.dir=checkpoints/pi0_fast_ur5e_usb_insertion_lora/usb_insertion_50demos/4999

Then run inference client:

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
    python scripts/remote_pi_inference_dual_cam.py \
        --policy_host localhost --policy_port 8000

---

## Step 11: (Optional) RL Token Pipeline on HPC

    cd /data/beegfs/home/saifi/rlt_ur5e
    VLA_CONFIG=pi0_fast_ur5e_usb_insertion_lora bash hpc/07_rl_token.sh pi0fast

---

## Checklist for New Task

- [ ] Collect 50+ demonstrations
- [ ] Push to HuggingFace (LeRobot format)
- [ ] Add 3 configs to config.py (pi0, pi0-FAST, pi0.5)
- [ ] IMPORTANT: pi0-FAST uses rank=16 on HPC (NOT rank=4)
- [ ] Compute norm stats for all 3
- [ ] Transfer dataset to HPC cache
- [ ] Create SLURM scripts (copy from peg insertion templates)
- [ ] Submit training jobs
- [ ] Resume if needed (--resume flag)
- [ ] Transfer checkpoints: HPC -> WSL2 -> Ubuntu workstation
- [ ] Deploy inference server
- [ ] Test with robot
- [ ] (Optional) Run RL Token pipeline on HPC
- [ ] (Optional) Online RL fine-tuning

---

## Quick Reference: Config Names

| Task | pi0 | pi0-FAST | pi0.5 |
|------|-----|----------|-------|
| Peg Insertion | pi0_ur5e_peg_insertion_lora | pi0_fast_ur5e_peg_insertion_lora | pi05_ur5e_peg_insertion_lora |
| USB Insertion | pi0_ur5e_usb_insertion_lora | pi0_fast_ur5e_usb_insertion_lora | pi05_ur5e_usb_insertion_lora |
| (Your Task) | pi0_ur5e_YOUR_TASK_lora | pi0_fast_ur5e_YOUR_TASK_lora | pi05_ur5e_YOUR_TASK_lora |

---

## LoRA Rank Reference (IMPORTANT)

| Model | VLM (PaliGemma 2B) | Action Expert (300M) | Notes |
|-------|--------------------|--------------------|-------|
| pi0 | rank=16, alpha=16 | rank=32, alpha=32 | Default from gemma_2b_lora |
| pi0-FAST (HPC) | rank=16, alpha=16 | N/A (FAST tokenizer) | Do NOT set rank=4 on HPC! |
| pi0-FAST (local 16GB) | rank=4, alpha=4 | N/A | Only for RTX 5070 Ti / 16GB GPUs |
| pi0.5 | rank=16, alpha=16 | rank=32, alpha=32 | Same as pi0 |

The base model checkpoint (gs://openpi-assets/checkpoints/pi0_fast_base/params)
has LoRA matrices at rank=16. If you train at rank=4, the dimensions still work
(LoRA is initialized fresh) but you get less expressiveness.

For RLT (RL Token) compatibility, all models should use the same rank
so the embedding dimensions are consistent for the encoder-decoder.
