# UR5e HIL-SERL

**Human-in-the-Loop Sample-Efficient Reinforcement Learning for UR5e + Robotiq Hand-E**

Learn precise robotic manipulation tasks (peg insertion, PCB assembly) in **under 1 hour** of real-world training using reinforcement learning with human corrections.

[![GitHub](https://img.shields.io/badge/GitHub-ur5e__hil__serl-blue)](https://github.com/saifi-shahrukh/ur5e_hil_serl)

---

## 🎥 What This Does

1. **You demonstrate** the task 20 times using keyboard teleoperation (~15 min)
2. **You train a reward classifier** to detect task completion (~5 min)
3. **RL trains autonomously** while you occasionally correct mistakes (~30-60 min)
4. **Result**: A policy that performs the task faster and more reliably than you can teleoperate

---

## 🖥️ Hardware Requirements

| Component | Model | Connection | Notes |
|-----------|-------|------------|-------|
| Robot Arm | **UR5e** | Ethernet (172.22.1.139) | Any UR e-series should work |
| Gripper | **Robotiq Hand-E** | UR Tool Port | Async Modbus TCP control |
| Wrist Camera | **Intel RealSense D435** | USB 3.0 | Serial: varies per unit |
| Overview Camera | **Kinect Xbox One v2** | USB 3.0 | Needs libfreenect2 |
| GPU | **NVIDIA RTX** (CUDA 12.x) | PCIe | For JAX training |
| Input | **Keyboard** | USB | For teleoperation (no SpaceMouse needed) |

---

## ⚡ Quick Start (5 minutes to first run)

### 1. Install

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl

# Create Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install packages
pip install -e .
pip install -e serl_launcher/
pip install --upgrade "jax[cuda12]==0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow tf-keras ml_collections matplotlib natsort wandb ur-rtde pyrealsense2 freenect2 pynput

# Verify GPU
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0)]
```

### 2. Record 20 Demonstrations

```bash
cd examples
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"
python record_demos.py --exp_name peg_insertion --successes_needed 20
```

**Controls**: Arrow keys = XY, `1`/`0` = Z up/down, Right Ctrl = gripper

### 3. Train Reward Classifier

```bash
# First collect success/failure images
python record_success_fail.py --exp_name peg_insertion --successes_needed 200

# Then train
python train_reward_classifier.py --exp_name peg_insertion --num_epochs 150
```

### 4. Train RL Policy (Two Terminals)

**Terminal 1 — Learner (GPU):**
```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples && source ../.venv/bin/activate && export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH" && python train_rlpd.py --exp_name peg_insertion --demo_path ./demo_data/YOUR_DEMO_FILE.pkl --learner
```

**Terminal 2 — Actor (CPU + Robot):**
```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples && source ../.venv/bin/activate && export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH" && python train_rlpd.py --exp_name peg_insertion --actor
```

**⚠️ Important**: Start the learner FIRST, wait for `"Filling up replay buffer"`, then start the actor.

### 5. Evaluate a Trained Policy

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples && source ../.venv/bin/activate && export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH" && python train_rlpd.py --exp_name peg_insertion --actor --checkpoint_path ./checkpoints/peg_insertion --eval_checkpoint_step 25000 --eval_n_trajs 20
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING SYSTEM                               │
├────────────────────────────┬────────────────────────────────────────┤
│   LEARNER (Terminal 1)     │          ACTOR (Terminal 2)            │
│   GPU — trains policy      │          CPU — controls robot          │
│                            │                                        │
│   ┌──────────────────┐     │     ┌──────────────────────────┐      │
│   │  SAC Agent       │     │     │  SAC Agent (copy)        │      │
│   │  (critic+actor)  │◄────┼─────│  (inference only)        │      │
│   └────────┬─────────┘     │     └──────────┬───────────────┘      │
│            │               │                 │                      │
│   ┌────────▼─────────┐     │     ┌───────────▼──────────────┐      │
│   │  Replay Buffer   │◄────┼─────│  Environment             │      │
│   │  (50% demo +     │     │     │  ┌────────────────────┐  │      │
│   │   50% online)    │     │     │  │ UR5eEnv            │  │      │
│   └──────────────────┘     │     │  │  ├─ Controller     │  │      │
│            │               │     │  │  │  (100Hz force)  │  │      │
│   ┌────────▼─────────┐     │     │  │  ├─ RealSense D435 │  │      │
│   │  WandB Logging   │     │     │  │  ├─ Kinect v2      │  │      │
│   └──────────────────┘     │     │  │  └─ Hand-E Gripper │  │      │
│                            │     │  └────────────────────┘  │      │
│   ZMQ Server :5588/:5589   │     │  ┌────────────────────┐  │      │
│   (weights broadcast)      │     │  │ Human Intervention │  │      │
│                            │     │  │ (keyboard control) │  │      │
│                            │     │  └────────────────────┘  │      │
└────────────────────────────┴─────┴──────────────────────────────────┘
```

### How It Works

1. **Actor** runs episodes on the real robot, sending transitions to the learner
2. **Learner** trains the SAC neural network on GPU using RLPD:
   - 50% samples from human demonstrations
   - 50% samples from online robot experience
3. **Learner broadcasts** updated weights to the actor every 50 steps
4. **Human intervenes** via keyboard when the robot makes mistakes
   - Corrections go into a separate intervention buffer
   - This is the "human-in-the-loop" part — crucial for fast learning

---

## 📊 Training Timeline

| Steps | Time (~8 Hz) | What Happens |
|-------|-------------|---------------|
| 0-100 | ~12 sec | Replay buffer fills (learner waits) |
| 100-1000 | ~2 min | Random/early exploration |
| 1000-5000 | ~10 min | Policy learns direction to target |
| 5000-15000 | ~25 min | Insertion attempts begin |
| 15000-30000 | ~50 min | Consistent success (reduce interventions) |
| 30000+ | 1+ hr | Near-perfect, faster execution |

**Checkpoints** saved every 5000 steps to `./checkpoints/peg_insertion/`

---

## 📁 Project Structure

```
ur5e_hil_serl/
├── examples/                        # ← All training scripts run from here
│   ├── train_rlpd.py               # Main training script (actor + learner)
│   ├── record_demos.py             # Step 1: Collect demonstrations
│   ├── record_success_fail.py      # Step 2: Collect classifier data
│   ├── train_reward_classifier.py  # Step 3: Train reward classifier
│   ├── train_bc.py                 # Behavioral cloning baseline
│   ├── train_hgdagger.py           # DAgger training
│   ├── run_training.sh             # Convenience launcher
│   └── experiments/                # Task configs
│       ├── peg_insertion/          # ← Primary task (UR5e)
│       │   ├── config.py           # All hardware + RL parameters
│       │   └── wrapper.py          # Task-specific env wrapper
│       ├── pcb_insertion/          # PCB assembly task
│       ├── cable_routing/          # Cable routing task
│       ├── bin_relocation/         # Pick-and-place task
│       └── mappings.py             # exp_name → config mapping
│
├── serl_robot_infra/               # Robot control layer
│   ├── robot_controllers/
│   │   └── ur5e_controller.py      # 100Hz impedance controller
│   └── ur_env/
│       ├── envs/
│       │   ├── ur5e_env.py         # Gymnasium environment
│       │   ├── wrappers.py         # Intervention, classifier, gripper
│       │   └── relative_env.py     # Actions in TCP frame
│       ├── camera/                 # RealSense + Kinect drivers
│       ├── spacemouse/             # Keyboard/SpaceMouse input
│       └── utils/                  # Gripper driver, rotations
│
├── serl_launcher/                  # RL algorithms & networks
│   └── serl_launcher/
│       ├── agents/continuous/      # SAC, hybrid SAC agents
│       ├── networks/               # Actor-critic, reward classifier
│       ├── vision/                 # ResNet-10, data augmentation
│       ├── wrappers/               # Obs wrappers, chunking
│       ├── data/                   # Replay buffers
│       └── utils/                  # Launcher, timer, logging
│
├── tests/                          # Unit & integration tests
├── docs/                           # 10 detailed documentation guides
├── QUICKSTART.md                   # Condensed setup guide
└── README_2.md                     # ForceMode singularity analysis
```

---

## 🔧 Configuration

All task parameters are in `examples/experiments/peg_insertion/config.py`:

```python
# Robot
ROBOT_IP = "172.22.1.139"
CONTROLLER_HZ = 100

# Poses (measure with teach pendant)
RESET_Q = np.deg2rad([33.56, -76.79, -132.20, -60.98, 90.22, 35.98])  # Start
TARGET_POSE = np.array([0.362, 0.080, 0.085, 2.176, -2.266, 0.0])     # Goal
HOME_Q = np.array([0.7854, -1.1868, -1.7802, -1.7453, 1.5708, 0.0])   # Safe

# Safety box (keeps robot away from singularities)
ABS_POSE_LIMIT_LOW  = np.array([0.28, -0.02, 0.03, -0.10, -0.10, -0.10])
ABS_POSE_LIMIT_HIGH = np.array([0.42,  0.14, 0.20,  0.10,  0.10,  0.10])

# Action scale: [position_m/step, rotation_rad/step, gripper]
ACTION_SCALE = np.array([0.005, 0.03, 1.0])
```

---

## 🔄 Resume vs. Fresh Training

| Scenario | What Happens |
|----------|-------------|
| First run (no checkpoints) | Starts from scratch |
| Checkpoints exist | Prompts "Press Enter to resume" → loads weights + buffers |
| Want fresh start | Delete `./checkpoints/peg_insertion/` folder first |
| Buffer data only (no model checkpoint) | Loads buffer data, trains new model |

---

## 🧠 Reward Classifier: Key Design Decisions

### The Problem
A binary reward classifier trained on success/failure images may fire **prematurely** when the robot is near (but not at) the goal. For peg insertion:
- Peg **above hole**: classifier probability ~0.60-0.68
- Peg **in hole**: classifier probability ~0.72-0.77

A simple threshold (e.g., >0.60) causes false positives. A strict threshold (e.g., >0.85) **never fires** because the classifier's max confidence is ~0.77.

### The Solution: Consecutive Frame Filtering

Instead of firing on a single frame exceeding the threshold, we require **N consecutive frames** above the threshold:

```python
# In config.py reward_func:
CONSECUTIVE_NEEDED = 3  # Require 3 frames in a row
THRESHOLD = 0.70        # Probability threshold

def reward_func(obs):
    prob = sigmoid(classifier_fn(obs))
    if prob > THRESHOLD:
        consecutive_count += 1
    else:
        consecutive_count = 0
    if consecutive_count >= CONSECUTIVE_NEEDED:
        return 1  # TRUE success
    return 0
```

**Why this works:**
- Peg passes *above* hole → maybe 1 frame >0.70 → streak resets → **no reward** ✅
- Peg truly *inserted* → stable 3+ frames >0.70 → **reward=1** ✅

### Debugging the Classifier

Test classifier on saved data to find the right threshold:
```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples && source ../.venv/bin/activate && \
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH" && python3 -c "
import jax, os, pickle as pkl
from jax import numpy as jnp
from serl_launcher.networks.reward_classifier import load_classifier_func
from experiments.mappings import CONFIG_MAPPING

config = CONFIG_MAPPING['peg_insertion']()
env = config.get_environment(fake_env=True, save_video=False, classifier=False)
classifier_fn = load_classifier_func(
    key=jax.random.PRNGKey(0), sample=env.observation_space.sample(),
    image_keys=config.classifier_keys, checkpoint_path=os.path.abspath('./classifier_ckpt/'))
sigmoid = lambda x: 1/(1+jnp.exp(-x))

data = pkl.load(open('./classifier_data/YOUR_SUCCESS_FILE.pkl', 'rb'))
for i in range(10):
    logit = classifier_fn(data[i]['observations']).squeeze()
    print(f'success[{i}]: prob={sigmoid(logit).item():.3f}')
"
```

### Threshold Selection Guide

| Classifier Accuracy | Recommended Threshold | Consecutive Frames |
|--------------------|----------------------|--------------------|
| >95% | 0.85 | 1 (none needed) |
| 85-95% | 0.75 | 2 |
| 75-85% | 0.70 | 3 |
| <75% | Retrain with more data | — |

### Distance-Based Fallback

For quick verification that the RL pipeline works (bypassing the classifier):
```bash
python train_rlpd.py --exp_name peg_insertion --actor --no_classifier
```
This gives reward=1 when TCP is within `REWARD_THRESHOLD` (10mm position, 3° rotation) of `TARGET_POSE`.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ZMQError: Address already in use (5588)` | `pkill -f train_rlpd.py` then retry |
| `FORCE MODE NOT POSSIBLE IN SINGULARITY` | Safety box too wide (already fixed in this repo) |
| `forceMode failed, recovering...` | Normal during exploration — reduces over time |
| `RTDE input registers already in use` | Kill other processes connected to robot |
| `CUDA_ERROR_NO_DEVICE` (actor) | Harmless warning — actor runs on CPU |
| Robot doesn't move | Check teach pendant for protective stops |
| Learner stuck at "Filling up replay buffer" | Actor isn't running or not connected |

**Kill everything and restart:**
```bash
pkill -f train_rlpd.py; lsof -ti:5588 | xargs kill -9 2>/dev/null; lsof -ti:5589 | xargs kill -9 2>/dev/null; echo "done"
```

---

## 📚 Documentation

Detailed guides in [`docs/`](docs/):

| # | Guide | Description |
|---|-------|-------------|
| 01 | [Environment Setup](docs/01_environment_setup.md) | System deps, CUDA, network |
| 02 | [UV Environment](docs/02_uv_environment_setup.md) | Python venv & packages |
| 03 | [Demo Collection](docs/03_demo_data_collection.md) | Recording demonstrations |
| 04 | [Reward Classifier](docs/04_reward_classifier.md) | Success detection training |
| 05 | [Actor Setup](docs/05_actor_setup.md) | Robot-side process |
| 06 | [Training Pipeline](docs/06_training_pipeline.md) | Full training walkthrough |
| 07 | [Evaluation](docs/07_evaluation.md) | Testing saved checkpoints |
| 08 | [UR Controller](docs/08_ur_controller.md) | Impedance control & forceMode |
| 09 | [UR5e Environment](docs/09_ur5e_environment.md) | Gym env & wrapper stack |
| 10 | [Examples Reference](docs/10_examples_reference.md) | Every script explained |

---

## 🏆 References

This project builds upon:

| Project | Paper | Role |
|---------|-------|------|
| [HIL-SERL](https://github.com/rail-berkeley/hil-serl) | [Science Robotics 2025](https://arxiv.org/abs/2410.21845) | Original framework (Franka) |
| [SERL](https://github.com/rail-berkeley/serl) | [arXiv 2401.16013](https://arxiv.org/abs/2401.16013) | Foundation algorithm |
| [Voxel-SERL](https://github.com/nisutte/voxel-serl) | [arXiv 2503.02405](https://arxiv.org/abs/2503.02405) | UR5 reference + 3D perception |
| [ur-rtde](https://sdurobotics.gitlab.io/ur_rtde/) | — | UR robot communication |

---

## 📜 Citation

```bibtex
@article{luo2025hilserl,
  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Luo, Jianlan and Xu, Charles and Wu, Jeffrey and Levine, Sergey},
  journal={Science Robotics},
  volume={10},
  number={105},
  year={2025}
}
```

---

## License

MIT License (same as [HIL-SERL](https://github.com/rail-berkeley/hil-serl))
