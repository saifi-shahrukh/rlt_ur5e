# 04 — Reward Classifier Setup, Training, and Usage

## Overview

The reward classifier is a binary image classifier (ResNet-10 backbone) that determines whether the task is complete (reward=1) or not (reward=0). It uses camera images to detect success.

## Step 1: Collect Classifier Training Data

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python record_success_fail.py --exp_name peg_insertion --successes_needed 200
```

### How it works:
- **Default**: All recorded frames are labeled as **negative** (failure/not-done)
- **Press SPACE**: While held, frames are labeled as **positive** (success/done)
- Teleoperate the robot normally
- When peg is **fully inserted**, hold SPACE and move slightly (~10 frames per insertion)
- Aim for ~200 positive frames across 10-15 insertions

### Tips for Robust Classifier:
- Collect **2-3x more negatives than positives** to prevent false positives
- Include "almost done" negatives (peg partially inserted but not fully)
- Include negatives from various positions in the workspace
- False positives are WORSE than false negatives for RL training

### Output:
```
./classifier_data/peg_insertion_<N>_success_images_<timestamp>.pkl
./classifier_data/peg_insertion_<N>_failure_images_<timestamp>.pkl
```

## Step 2: Train the Classifier

```bash
python train_reward_classifier.py --exp_name peg_insertion --num_epochs 150
```

### Parameters:
- `--num_epochs`: Number of training epochs (default: 150)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 3e-4)

### Output:
```
./classifier_ckpt/checkpoint_150/
```

### Expected Results:
- Accuracy: ~75-85% after 150 epochs
- The classifier is binary (inserted vs. not inserted)
- Higher accuracy is better, but even 75% works for RL training

## Step 3: Usage in Training

The classifier is automatically loaded during RL training (unless `--no_classifier` is passed):

```python
# In peg_insertion/config.py:
classifier_fn = load_classifier_func(
    key=jax.random.PRNGKey(0),
    sample=env.observation_space.sample(),
    image_keys=["wrist_1", "overview"],
    checkpoint_path="classifier_ckpt/",
)

# Reward function:
def reward_func(obs):
    logit = classifier_fn(obs)
    return int(sigmoid(logit) > 0.85)  # Binary reward
```

## Alternative: Distance-Based Reward

If you don't want to train a classifier, use distance-based reward:

```bash
python train_rlpd.py --exp_name peg_insertion --no_classifier --learner
python train_rlpd.py --exp_name peg_insertion --no_classifier --actor
```

This compares the TCP pose to `TARGET_POSE` and gives reward=1 when within `REWARD_THRESHOLD`.

## Retraining the Classifier

If the classifier isn't working well:
1. Collect more diverse data (especially hard negatives)
2. Check image crops in config — ensure task-relevant area is visible
3. Train for more epochs
4. Consider using a separate/additional camera for classification
