# 06 — Learning / Training Pipeline Setup

## Overview

HIL-SERL uses an **actor-learner** architecture:
- **Learner** (Terminal 1): Runs on GPU, trains the SAC neural network
- **Actor** (Terminal 2): Runs on CPU, controls the robot, sends experience

They communicate via ZMQ on ports 5588 (request/reply) and 5589 (broadcast).

## Architecture

```
┌─────────────────────┐         ZMQ (5588/5589)        ┌─────────────────────┐
│      LEARNER        │◄──────────────────────────────►│       ACTOR          │
│  (GPU, training)    │    weights / transitions       │  (CPU, robot)        │
│                     │                                │                      │
│  SAC Agent          │                                │  SAC Agent (copy)    │
│  Demo Buffer (50%)  │                                │  UR5e Environment    │
│  Online Buffer (50%)│                                │  Human Intervention  │
│  WandB Logging      │                                │  Camera Capture      │
└─────────────────────┘                                └─────────────────────┘
```

## Step-by-Step

### 1. Start the Learner (Terminal 1)

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python train_rlpd.py --exp_name peg_insertion \
    --demo_path ./demo_data/peg_insertion_608_transitions_2026-05-04_17-26-14.pkl \
    --learner
```

Wait until you see: `"Filling up replay buffer"` — this means the learner is ready.

### 2. Start the Actor (Terminal 2)

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python train_rlpd.py --exp_name peg_insertion --actor
```

### 3. Training Begins

Once the actor sends enough transitions (100 by default), the learner starts training:
```
Filling up replay buffer: 101it [01:19, 1.28it/s]
sent initial network to actor
learner: 0%| | 553/974999 [01:26<25:27:23, 10.63it/s]
```

## Training Algorithm: RLPD

**Reinforcement Learning with Prior Data** — each batch is:
- 50% from **demo buffer** (human demonstrations)
- 50% from **online buffer** (actor's experience)

This ensures the policy always has good examples to learn from.

## Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Training batch size |
| `training_starts` | 100 | Min transitions before training |
| `cta_ratio` | 4 | Critic-to-actor update ratio |
| `steps_per_update` | 50 | Steps between weight broadcasts |
| `checkpoint_period` | 5000 | Steps between checkpoint saves |
| `discount` | 0.97 | RL discount factor |
| `max_steps` | 1,000,000 | Maximum training steps |

## Resuming Training

If checkpoints exist in `./checkpoints/peg_insertion/`, the script will:
1. Prompt: "Checkpoint path already exists. Press Enter to resume training."
2. Load the latest checkpoint
3. Resume from where it left off

## Starting Fresh

```bash
mv ./checkpoints/peg_insertion ./checkpoints/peg_insertion_backup
```

## Monitoring (WandB)

Metrics are logged to Weights & Biases:
- `episode/return` — should increase over time
- `episode/intervention_count` — should decrease
- `episode/length` — should decrease (faster completion)
- `critic_loss` — should decrease
- `actor_loss` — should stabilize

View at: https://wandb.ai/saifi/hil-serl

## Expected Timeline

| Phase | Steps | Time (~10 it/s) | What Happens |
|-------|-------|-----------------|---------------|
| Random exploration | 0-1000 | ~2 min | Policy outputs random actions |
| Early learning | 1000-5000 | ~8 min | Movement becomes purposeful |
| Task attempts | 5000-15000 | ~25 min | Starts attempting insertion |
| Convergence | 15000-30000 | ~50 min | Consistent success |
| Refinement | 30000+ | 1+ hr | Near-perfect, faster execution |

## Convenience Script

```bash
./run_training.sh learner   # Terminal 1
./run_training.sh actor     # Terminal 2
./run_training.sh kill      # Kill leftover processes
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ZMQError: Address already in use (5588)` | `lsof -ti:5588 \| xargs kill -9` |
| Learner stuck at "Filling up replay buffer" | Actor isn't running or isn't connected |
| `NotImplementedError: Must be either a learner or an actor` | Missing `--learner` or `--actor` flag |
| Training very slow | Check GPU usage with `nvidia-smi` |
