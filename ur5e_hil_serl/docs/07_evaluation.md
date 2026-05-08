# 07 — Evaluation on Real Hardware Using Saved Checkpoints

## Overview

After training, you can evaluate the learned policy on the real robot without the learner running. The actor loads a specific checkpoint and runs deterministic rollouts.

## Running Evaluation

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"

python train_rlpd.py --exp_name peg_insertion \
    --actor \
    --checkpoint_path ./checkpoints/peg_insertion \
    --eval_checkpoint_step 25000 \
    --eval_n_trajs 20
```

### Parameters:
- `--checkpoint_path`: Directory containing the checkpoint files
- `--eval_checkpoint_step`: Which checkpoint step to load (e.g., 25000)
- `--eval_n_trajs`: Number of evaluation episodes to run

## What Happens During Evaluation

1. Loads the specified checkpoint
2. Runs `eval_n_trajs` episodes with the policy (no exploration noise)
3. Records success rate and average completion time
4. Prints results:
   ```
   success rate: 0.85
   average time: 4.2
   ```

## Available Checkpoints

```bash
ls ./checkpoints/peg_insertion/
# checkpoint_5000  checkpoint_10000  checkpoint_15000  checkpoint_20000  checkpoint_25000
```

## Comparing Checkpoints

Run evaluation on multiple checkpoints to see learning progression:

```bash
for step in 5000 10000 15000 20000 25000; do
    echo "=== Evaluating step $step ==="
    python train_rlpd.py --exp_name peg_insertion \
        --actor \
        --checkpoint_path ./checkpoints/peg_insertion \
        --eval_checkpoint_step $step \
        --eval_n_trajs 10
    echo ""
done
```

## Recording Video

```bash
python train_rlpd.py --exp_name peg_insertion \
    --actor \
    --checkpoint_path ./checkpoints/peg_insertion \
    --eval_checkpoint_step 25000 \
    --eval_n_trajs 5 \
    --save_video
```

Videos are saved to the experiment directory.

## Expected Results

| Training Steps | Expected Success Rate |
|---------------|----------------------|
| 5,000 | 0-20% |
| 10,000 | 20-50% |
| 15,000 | 50-80% |
| 25,000 | 80-95% |
| 50,000+ | 95-100% |

Results vary depending on task difficulty, demo quality, and intervention frequency.

## Deploying Without Learner

For deployment, only the actor is needed. The learner can be shut down after training completes.
