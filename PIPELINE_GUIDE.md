# Complete Pipeline: VLA Training to Inference to RL Token to Online RL

## Current Training Status (2026-05-12)

| Model | Steps | Status | Checkpoint Step | Rate |
|-------|-------|--------|-----------------|------|
| pi0.5 | 5000/5000 | COMPLETE | 4999 | 10.1s/step |
| pi0 | ~4640/5000 | Needs Resume | 4000 | 9.3s/step |
| pi0-FAST | ~4040/5000 | Needs Resume | 4000 | 10.6s/step |

Checkpoint base path on HPC:
  /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/

---

## Phase 1: Complete Remaining Training (on HPC)

### Step 1.1: Resume pi0 and pi0-FAST

SSH to HPC headnode and run:

  cd /data/beegfs/home/saifi/rlt_ur5e
  bash hpc/04_resume.sh both

This submits two SLURM jobs:
  - pi0: ~360 steps remaining, about 1 hour
  - pi0-FAST: ~960 steps remaining, about 3 hours

Or individually:
  bash hpc/04_resume.sh pi0
  bash hpc/04_resume.sh pi0fast

The KEY difference from initial training: uses --resume flag instead of --overwrite.
This tells OpenPI to load the latest checkpoint and continue from there.

### Step 1.2: Monitor progress

  squeue -u saifi
  tail -f /data/beegfs/home/saifi/logs/pi0_resume_*.err
  tail -f /data/beegfs/home/saifi/logs/pi0fast_resume_*.err

Also check W&B: https://wandb.ai/saifi/openpi

### Step 1.3: Verify completion

After jobs finish, verify all 3 have final checkpoints:

  ls -d /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/*/peg_insertion_50demos/4999

Expected output:
  .../pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4999
  .../pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999
  .../pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

---

## Phase 2: Download Checkpoints to Robot Workstation

Run from your LOCAL robot workstation (not HPC):

  cd ~/ur5e_hande_workspace/rlt_ur5e
  bash hpc/05_download_checkpoints.sh

Choose option 5 for params-only (smallest download, sufficient for inference).
Or option 1 for full checkpoints (needed if you want to resume training later).

Manual rsync (if script not available locally):

  rsync -avz --progress \
    saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/ \
    ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/

---

## Phase 3: VLA Inference (Serve Policy)

On the robot workstation (needs GPU):

  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
  source .venv/bin/activate

### Serve pi0-FAST (recommended - fastest inference):

  python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir=checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

### Serve pi0:

  python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=pi0_ur5e_peg_insertion_lora \
    --policy.dir=checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

### Serve pi0.5:

  python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_ur5e_peg_insertion_lora \
    --policy.dir=checkpoints/pi05_ur5e_peg_insertion_lora/peg_insertion_50demos/4999

The server listens on ws://0.0.0.0:8000 and accepts WebSocket connections.

---

## Phase 4: Test Inference with Robot

Run the inference client (from lerobot_ur5e_gello):

  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
  source .venv/bin/activate

  python scripts/remote_pi_inference_dual_cam.py \
    --policy_host localhost \
    --policy_port 8000

This connects to the policy server and runs the robot.

---

## Phase 5: Extract VLM Embeddings (for RL Token Training)

This step extracts the internal VLM representations from the trained VLA.
Run in the openpi venv on a machine with GPU:

  cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
  source .venv/bin/activate
  cd ~/ur5e_hande_workspace/rlt_ur5e

  python rlt/training/extract_embeddings.py \
    --config_name pi0_fast_ur5e_peg_insertion_lora \
    --checkpoint_dir openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_50demos/4999 \
    --output checkpoints/rl_token/embeddings_peg_insertion.pt \
    --n_samples 200

Output: checkpoints/rl_token/embeddings_peg_insertion.pt
  Contains: list of (N_prefix, 2048) tensors - one per demo frame

---

## Phase 6: Train RL Token Encoder-Decoder

This trains a small transformer to compress VLM embeddings into a compact
RL token. Run in the hilserl venv (PyTorch):

  cd ~/ur5e_hande_workspace/rlt_ur5e
  source ur5e_hil_serl/.venv/bin/activate

  python -m rlt.training.train_rl_token \
    --cache checkpoints/rl_token/embeddings_peg_insertion.pt \
    --save_path checkpoints/rl_token/peg_insertion_v1.pt \
    --token_dim 512 \
    --steps 5000 \
    --batch_size 32

Output: checkpoints/rl_token/peg_insertion_v1.pt
  Contains: model state_dict + config + best loss

For quick testing without VLA (synthetic data):

  python -m rlt.training.train_rl_token \
    --synthetic \
    --save_path checkpoints/rl_token/test_synthetic.pt \
    --steps 2000

---

## Phase 7: Online RL with RL Token (Future)

Once the RL Token model is trained:

1. The VLA serves as the base policy (Phase 3)
2. The RL Token encoder compresses VLM state into z_rl (512-dim)
3. A small SAC actor-critic trains on z_rl + proprioception
4. The SAC correction is added to VLA actions

This is the full RLT pipeline from the Physical Intelligence paper.
See rlt/examples/peg_insertion/train_rlt.py for the implementation.

---

## Troubleshooting

### Resume fails with "Cannot resume and overwrite at the same time"

The scripts use --resume (not --overwrite). If the checkpoint dir has a
wandb_id.txt but no valid checkpoints, delete the dir and start fresh:

  rm -rf checkpoints/<config>/peg_insertion_50demos/
  # Then use the original training script with --overwrite

### Resume fails with "wandb_id.txt not found"

If W&B tracking is broken during resume, set offline mode:

  export WANDB_MODE=offline

Or add to the SLURM script before the python command.

### OOM on pi0.5

pi0.5 needs batch_size=4 with grad_accumulation_steps=2.
batch_size=8 triggers XLA rematerialization and exceeds V100 32GB.

### Memory settings

pi0.5 uses XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 (not 0.95) to leave
headroom for rematerialization.

---

## File Locations Summary

### On HPC:
  Project:     /data/beegfs/home/saifi/rlt_ur5e/
  OpenPI:      /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/
  Checkpoints: /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/
  Logs:        /data/beegfs/home/saifi/logs/
  Dataset:     ~/.cache/huggingface/lerobot/saifi/ur5e-peg-insertion-50demos-v2/

### On Robot Workstation:
  Project:     ~/ur5e_hande_workspace/rlt_ur5e/
  OpenPI:      ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/
  Checkpoints: ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/
  RL Token:    ~/ur5e_hande_workspace/rlt_ur5e/checkpoints/rl_token/

### Scripts:
  hpc/03_train.sh          - Submit fresh training (all 3 models)
  hpc/04_resume.sh         - Resume interrupted training
  hpc/05_download_checkpoints.sh - Download to local machine
  scripts/start_vla_server.sh    - Start inference server
  rlt/training/extract_embeddings.py   - Extract VLM embeddings
  rlt/training/train_rl_token.py       - Train RL Token model
  rlt/examples/peg_insertion/train_rlt.py - Full online RL pipeline
