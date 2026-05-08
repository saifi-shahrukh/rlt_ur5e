# Commands Reference

All commands for the RLT-UR5e pipeline.

---

## 1. Demo Collection

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

python scripts/record.py \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 \
    --robot.freedrive=False \
    --teleop.type=keyboard_ur5e \
    --teleop.robot_ip=172.22.1.139 \
    --teleop.mode=cartesian \
    --teleop.trans_vel=0.08 \
    --dataset.repo_id=saifi/ur5e-peg-insertion-dual \
    --dataset.single_task="Pick up the peg and insert it into the hole." \
    --dataset.root=/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/datasets \
    --dataset.num_episodes=50 \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=False \
    --dataset.video=True

# Resume existing dataset:
# Add: --resume
```

---

## 2. VLA Fine-tuning (Local — π0-FAST)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e

# Compute norm stats (once per dataset change)
.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_fast_ur5e_peg_insertion_lora

# Train (background, ~4h)
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_autotune_level=0"

.venv/bin/python scripts/train.py pi0_fast_ur5e_peg_insertion_lora \
    --exp-name=peg_insertion_9demos --overwrite

# Monitor:
tail -f /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/training_log.txt
nvidia-smi
```

---

## 3. VLA Fine-tuning (HPC — π0/π0.5)

See [cluster.md](cluster.md) for full details.

```bash
# On HPC:
cd /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e
sbatch scripts/train_hpc_pi0.sh     # π0
sbatch scripts/train_hpc_pi05.sh    # π0.5
```

---

## 4. VLA Server (must run before inference or RLT)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e

.venv/bin/python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi0_fast_ur5e_peg_insertion_lora \
    --policy.dir checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_9demos/29999
```

---

## 5. VLA-Only Inference (Baseline)

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/robolab-2/freenect2/lib:${LD_LIBRARY_PATH:-}

python scripts/remote_pi_inference_dual_cam.py \
    --ip=localhost --port=8000 \
    --prompt="Pick up the peg and insert it into the hole." \
    --robot.type=ur5e_dual_cam \
    --robot.ip=172.22.1.139 --fps=30
```

---

## 6. RL Token Training

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Extract VLA embeddings from checkpoint
python -m rlt.models.extract_embeddings \
    --checkpoint openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora/peg_insertion_9demos/29999 \
    --config pi0_fast_ur5e_peg_insertion_lora \
    --dataset saifi/ur5e-peg-insertion-dual \
    --output checkpoints/rl_token/embeddings_peg_insertion_9demos.pt

# Train RL Token model
python -m rlt.models.train_rl_token \
    --embeddings checkpoints/rl_token/embeddings_peg_insertion_9demos.pt \
    --output checkpoints/rl_token/peg_insertion_9demos_v1.pt \
    --token_dim 512 --epochs 2000
```

---

## 7. RLT Online RL Training

```bash
cd /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
source ur5e_hil_serl/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"
export JAX_PLATFORMS=cpu

# Test with fake env (no hardware needed)
python -m rlt.examples.peg_insertion.train_rlt --fake_env

# VLA-only warmup (safe)
python -m rlt.examples.peg_insertion.train_rlt --warmup_only

# Full RLT training (VLA server must be running in another terminal)
python -m rlt.examples.peg_insertion.train_rlt
```

---

## 8. Evaluation

```bash
# With RLT residual corrections
python -m rlt.examples.peg_insertion.train_rlt \
    --eval_only --eval_episodes 20 \
    --checkpoint checkpoints/rlt_runs/peg_insertion/best.pkl

# VLA-only (zero residual) for comparison
python -m rlt.examples.peg_insertion.train_rlt \
    --eval_only --eval_episodes 20 --no_residual
```

---

## 9. Switching VLA Models

To change from π0-FAST to π0 or π0.5:

1. Update config:
   ```python
   # In rlt/examples/peg_insertion/config.py:
   vla_config_name: str = "pi0_ur5e_peg_insertion_lora"  # or pi05_...
   vla_checkpoint_dir: str = "path/to/pi0/checkpoint/29999"
   ```

2. Re-extract embeddings + re-train RL Token (VLA embeddings differ per model)
3. Serve the new model: change `--policy.config` and `--policy.dir`

---

## 10. Utility Commands

```bash
# Move robot to home position
python -c "
import rtde_control, numpy as np
rtc = rtde_control.RTDEControlInterface('172.22.1.139')
q = np.deg2rad([33.56, -76.79, -132.20, -60.98, 90.22, 35.98]).tolist()
rtc.moveJ(q, 0.5, 0.5); rtc.stopScript(); print('✓ Home')
"

# Check GPU
nvidia-smi

# Kill zombie processes
nvidia-smi | grep python  # Find PIDs
kill -9 <PID>
```
