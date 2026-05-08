# 10 — Examples Folder Reference

Detailed documentation of every script and file in the `examples/` directory.

---

## Python Scripts

### `train_rlpd.py` — Main RL Training Script

**Purpose**: Unified entry point for both actor and learner in HIL-SERL training.

**Usage**:
```bash
# Learner (GPU)
python train_rlpd.py --exp_name peg_insertion --demo_path ./demo_data/FILE.pkl --learner

# Actor (CPU + Robot)
python train_rlpd.py --exp_name peg_insertion --actor

# Evaluation
python train_rlpd.py --exp_name peg_insertion --actor --checkpoint_path ./checkpoints/peg_insertion --eval_checkpoint_step 25000 --eval_n_trajs 20
```

**Flags**:
| Flag | Type | Description |
|------|------|-------------|
| `--exp_name` | string | Task name (maps to config in experiments/) |
| `--learner` | bool | Run as learner |
| `--actor` | bool | Run as actor |
| `--demo_path` | string (multi) | Path(s) to demo pkl files |
| `--checkpoint_path` | string | Path to save/load checkpoints |
| `--eval_checkpoint_step` | int | Checkpoint step to evaluate |
| `--eval_n_trajs` | int | Number of evaluation episodes |
| `--no_classifier` | bool | Use distance reward instead of classifier |
| `--debug` | bool | Disable WandB logging |
| `--ip` | string | Learner IP address (default: localhost) |
| `--seed` | int | Random seed (default: 42) |

---

### `record_demos.py` — Demonstration Collection

**Purpose**: Teleoperate the robot to collect successful task demonstrations.

**Usage**:
```bash
python record_demos.py --exp_name peg_insertion --successes_needed 20
```

**Inputs**: Keyboard control, camera observations, robot state
**Outputs**: `./demo_data/peg_insertion_<N>_transitions_<timestamp>.pkl`

**How it works**:
1. Creates the task environment with KeyboardIntervention wrapper
2. Human teleoperates until success (detected by classifier) or timeout
3. Successful episodes are saved; failed ones are discarded
4. Terminates after `successes_needed` successful demos collected

---

### `record_success_fail.py` — Classifier Data Collection

**Purpose**: Collect labeled images for training the reward classifier.

**Usage**:
```bash
python record_success_fail.py --exp_name peg_insertion --successes_needed 200
```

**Inputs**: Keyboard control + SPACE bar for labeling
**Outputs**:
- `./classifier_data/peg_insertion_<N>_success_images_<timestamp>.pkl`
- `./classifier_data/peg_insertion_<N>_failure_images_<timestamp>.pkl`

**How it works**:
1. All frames recorded by default as "failure" (negative)
2. While SPACE is held → frames labeled as "success" (positive)
3. Terminates after `successes_needed` positive frames

---

### `train_reward_classifier.py` — Train Binary Classifier

**Purpose**: Train a ResNet-10 binary classifier on success/failure images.

**Usage**:
```bash
python train_reward_classifier.py --exp_name peg_insertion --num_epochs 150
```

**Inputs**: Classifier data from `./classifier_data/`
**Outputs**: `./classifier_ckpt/checkpoint_<epoch>/`

**Architecture**: ResNet-10 (pretrained on ImageNet) → binary logit

---

### `train_bc.py` — Behavioral Cloning (Baseline)

**Purpose**: Train a policy via pure imitation learning (no RL).

**Usage**:
```bash
python train_bc.py --exp_name peg_insertion --demo_path ./demo_data/FILE.pkl
```

**Inputs**: Demo pkl file
**Outputs**: BC checkpoints

Useful as a baseline to compare against RL performance.

---

### `train_hgdagger.py` — Human-Guided DAgger Training

**Purpose**: Interactive imitation learning with online corrections.

**Usage**:
```bash
python train_hgdagger.py --exp_name peg_insertion --demo_path ./demo_data/FILE.pkl
```

DAgger iteratively collects corrections from a human expert and retrains.

---

## Shell Scripts

### `run_training.sh` — Convenience Launcher

**Purpose**: One-command launcher for learner, actor, or cleanup.

**Usage**:
```bash
./run_training.sh learner   # Start learner
./run_training.sh actor     # Start actor
./run_training.sh kill      # Kill processes and free ports
```

---

## Experiment Configs (`experiments/`)

### `experiments/mappings.py`
Maps `--exp_name` string to the corresponding config class.

### `experiments/config.py`
Base `DefaultTrainingConfig` class with default RL hyperparameters.

### `experiments/peg_insertion/config.py`
Full hardware + RL config for the peg insertion task. Contains:
- Robot IP, camera serials
- RESET_Q, HOME_Q, TARGET_POSE
- Safety box limits
- Action scaling
- Reward thresholds
- Environment factory (`get_environment()`)

### `experiments/peg_insertion/wrapper.py`
Task-specific `PegInsertionEnv` class wrapping `UR5eEnv` with:
- Custom reset logic (HOME_Q first, then RESET_Q)
- Task-specific reward shaping
- Episode termination conditions

### `experiments/peg_insertion/run_actor.sh` / `run_learner.sh`
Legacy shell scripts for launching actor/learner (from hil-serl convention).

---

## Per-Task Experiment Folders

| Task | Status | Description |
|------|--------|-------------|
| `peg_insertion/` | ✅ Active | Peg insertion with Hand-E (primary task) |
| `pcb_insertion/` | 🔧 Configured | PCB component insertion |
| `bin_relocation/` | 🔧 Configured | Object pick-and-place |
| `cable_routing/` | 🔧 Configured | Cable routing task |
| `ram_insertion/` | 📋 Reference | Original hil-serl Franka task |
| `usb_pickup_insertion/` | 📋 Reference | Original hil-serl Franka task |
| `egg_flip/` | 📋 Reference | Original hil-serl Franka task |
| `object_handover/` | 📋 Reference | Original hil-serl Franka task |

---

## Data Directories (gitignored)

| Directory | Contents |
|-----------|----------|
| `demo_data/` | Human demonstration pkl files |
| `classifier_data/` | Success/failure image pkl files |
| `classifier_ckpt/` | Trained reward classifier checkpoints |
| `checkpoints/` | RL training checkpoints |
