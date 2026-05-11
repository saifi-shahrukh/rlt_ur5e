# Adding a New Dataset for Fine-tuning

This guide explains how to add a new robot demonstration dataset for training
π0/π0.5/π0-FAST models on the HPC cluster.

---

## Overview

The training pipeline expects data in **LeRobot v2.0 format** (parquet + videos).
You need to:
1. Record demonstrations and convert to LeRobot format
2. Transfer dataset to HPC
3. Create a training config
4. Compute normalization statistics
5. Train

---

## Step 1: Prepare Dataset in LeRobot Format

Your dataset directory should look like:

    your-dataset-name/
    ├── data/
    │   └── train/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    ├── meta/
    │   ├── info.json              # dataset metadata
    │   ├── episodes.jsonl         # episode metadata
    │   └── tasks.jsonl            # task descriptions
    └── videos/
        ├── observation.images.camera1/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.camera2/
            ├── episode_000000.mp4
            └── ...

### info.json Structure

    {
      "codebase_version": "v2.0",
      "robot_type": "ur5e",
      "fps": 30,
      "features": {
        "observation.state": {"dtype": "float32", "shape": [7]},
        "action": {"dtype": "float32", "shape": [7]},
        "observation.images.overview_cam": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.wrist_cam": {"dtype": "video", "shape": [480, 640, 3]}
      }
    }

### Parquet Columns

Each episode parquet file should contain:
-                     — joint positions (float32 array)
-          — target joint positions (float32 array)
-                                  — frame indices referencing videos
-             — time in seconds
-                 — which episode
-               — frame number within episode

### Converting from Custom Format

If you have raw recordings, use lerobot's conversion tools:

    from lerobot.scripts.push_dataset_to_hub import (
        push_dataset_to_hub
    )

Or record directly with lerobot:

    python -m lerobot.scripts.control_robot record \
        --robot-path=... \
        --repo-id=your-name/dataset-name \
        --num-episodes=20

---

## Step 2: Transfer Dataset to HPC

From your local machine:

    # Create directory on HPC
    ssh saifi@hpc-headnode.iis.fhg.de \
        "mkdir -p /data/beegfs/home/saifi/datasets/your-name/your-dataset-name"

    # Transfer dataset
    rsync -avP --info=progress2 \
        /path/to/your-dataset-name/ \
        saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/datasets/your-name/your-dataset-name/

Then on HPC, create the symlink:

    mkdir -p ~/.cache/huggingface/lerobot/your-name
    ln -sf /data/beegfs/home/saifi/datasets/your-name/your-dataset-name \
           ~/.cache/huggingface/lerobot/your-name/your-dataset-name

---

## Step 3: Create Training Config

Edit the config file:

    openpi_ur5e/openpi-ur5e/src/openpi/training/config.py

Add a new TrainConfig at the end of the CONFIGS list. Use the existing peg insertion
config as a template:

    TrainConfig(
        name="pi0_your_task_name_lora",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,  # how many future actions to predict
        ),
        data=LeRobotUR5DualCamDataConfig(
            repo_id="your-name/your-dataset-name",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(asset_id="your-name/your-dataset-name"),
            extra_delta_transform=True,  # True for joint position control
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
        batch_size=4,         # adjust based on VRAM
        lr=2.5e-5,           # good default for LoRA
    ),

### Key Parameters to Adjust

| Parameter | What it Does | Guidance |
|-----------|--------------|----------|
| action_horizon | Future actions predicted | 30 for 30Hz (1 second lookahead) |
| num_train_steps | Total training iterations | 20k-50k depending on dataset size |
| batch_size | Samples per step | 4 for V100 16GB, 8-16 for 32GB |
| lr | Learning rate | 2.5e-5 (standard for LoRA) |
| keep_period | Checkpoint interval | 5000 (saves every 5k steps) |
| extra_delta_transform | Delta vs absolute actions | True for relative control |

### Camera Configuration

If your robot has different cameras, modify the repack transform in the data config:

    # In LeRobotUR5DualCamDataConfig or a new data config class:
    RepackTransform(structure={
        'observation/exterior_image_1_left': ('observation.images.your_camera1',),
        'observation/wrist_image_left': ('observation.images.your_camera2',),
        'observation/joint_position': ('observation.state',),
        'actions': ('action',),
        'prompt': ('task',)
    })

### Single Camera Setup

If you only have one camera, use                     instead of
                              and adjust the repack transform accordingly.

---

## Step 4: Compute Normalization Statistics

Norm stats must be pre-computed and stored in the assets directory.
Run on HPC (headnode is fine — no GPU needed):

    cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e

    # Source environment
    source .venv/activate_hpc.sh

    # Compute norm stats for your new config
    run_python scripts/compute_norm_stats.py --config-name=pi0_your_task_name_lora

This creates:

    assets/pi0_your_task_name_lora/your-name/your-dataset-name/
    └── norm_stats.json

Commit these to the repo:

    git add assets/
    git commit -m "Add norm stats for your_task_name"
    git push

---

## Step 5: Create SLURM Script

Copy an existing SLURM script and modify:

    cp hpc/slurm/pi0.sh hpc/slurm/pi0_your_task.sh

Edit the CONFIG and EXP_NAME:

    CONFIG="pi0_your_task_name_lora"
    EXP_NAME="your_task_20demos"

Add to 03_train.sh if desired, or submit directly:

    sbatch hpc/slurm/pi0_your_task.sh

---

## Step 6: Train and Monitor

    # Submit
    sbatch hpc/slurm/pi0_your_task.sh

    # Monitor
    squeue -u saifi
    tail -f /data/beegfs/home/saifi/logs/pi0_peg_<JOBID>.out

    # W&B
    # https://wandb.ai/saifi/openpi

---

## Tips for Good Results

### Dataset Quality
- **More demos = better**: 9 is minimum, 20-50 is good, 100+ is great
- **Consistent demonstrations**: similar speed, same strategy
- **Good camera coverage**: ensure task-relevant features are visible
- **Clean gripper signal**: binary open/close works best

### Hyperparameter Tuning
- Start with defaults (lr=2.5e-5, batch=4, 30k steps)
- If loss plateaus early: increase lr to 5e-5
- If loss is unstable: decrease lr to 1e-5
- If overfitting (loss drops to ~0 fast): reduce steps to 10-20k
- More demos: can increase batch size and steps

### Model Selection
- **π0-FAST**: Start here — trains fastest, good baseline
- **π0**: Better quality, standard choice
- **π0.5**: Best quality but needs more VRAM, try if π0 isn't good enough

### Evaluation
After training, deploy the checkpoint with the inference server:

    # On a machine with GPU:
    python scripts/serve_policy.py --config=pi0_your_task_name_lora \
        --checkpoint=checkpoints/pi0_your_task_name_lora/your_task_20demos/30000

---

## Example: Adding a New UR5e Task

Say you recorded 15 demos of "UR5e stacking blocks" with 2 cameras:

    # 1. Transfer data
    rsync -avP block_stacking/ saifi@hpc:datasets/saifi/ur5e-block-stacking/

    # 2. Add config (in config.py)
    TrainConfig(
        name="pi0_ur5e_block_stacking_lora",
        ...
        data=LeRobotUR5DualCamDataConfig(
            repo_id="saifi/ur5e-block-stacking",
            ...
        ),
        num_train_steps=40_000,  # more demos = more steps
        batch_size=4,
    ),

    # 3. Create symlink on HPC
    ln -sf /data/beegfs/home/saifi/datasets/saifi/ur5e-block-stacking \
           ~/.cache/huggingface/lerobot/saifi/ur5e-block-stacking

    # 4. Compute norm stats
    run_python scripts/compute_norm_stats.py --config-name=pi0_ur5e_block_stacking_lora

    # 5. Submit training
    sbatch hpc/slurm/pi0_your_task.sh
