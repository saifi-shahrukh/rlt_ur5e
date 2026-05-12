# Local GPU Training

For experimenting on your local GPU (RTX 5070 Ti 16GB) before committing to HPC.

## When to Use Local vs HPC

| Scenario | Local (16GB) | HPC (V100 32GB) |
|----------|-------------|----------------|
| Quick sanity check (100 steps) | YES | overkill |
| Test new config works | YES | overkill |
| Full 5000-step training | NO (too slow) | YES |
| Multi-task training | NO | YES (28 GPUs) |
| RL Token extraction | YES (if checkpoint fits) | YES |

## Local Training Configs

For 16GB GPU, use these settings:
- batch_size=1
- grad_accumulation_steps=8 (effective batch=8)
- pi0-FAST with paligemma_lora_rank=4 (saves ~3.7 GiB)

## Quick Test (100 steps)

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
    source .venv/bin/activate

    # Set memory-saving env vars
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true"

    # Quick test (pi0-FAST, 100 steps)
    python scripts/train.py pi0_fast_ur5e_peg_insertion_lora \
        --exp-name=local_test \
        --overwrite \
        --batch-size=1 \
        --grad-accumulation-steps=8 \
        --num-workers=2 \
        --num-train-steps=100 \
        --save-interval=50

## Scripts in This Directory

- train_local_pi0fast.sh  - Train pi0-FAST locally (rank=4, batch=1)
- train_local_test.sh     - Quick 100-step test run

## Note on LoRA Rank

The local configs use rank=4 for pi0-FAST to fit in 16GB.
For production training on HPC, use rank=16 (default).
See docs/NEW_TASK_GUIDE.md for details.
