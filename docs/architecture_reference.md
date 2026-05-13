# RLT-UR5e Architecture & Setup Reference

## 1. Current Configuration (Verified Working)

### VLA Checkpoints (in openpi_ur5e/openpi-ur5e/checkpoints/)

| Model | Config Name | Dir | Rank | Step | Status |
|-------|-------------|-----|------|------|--------|
| pi0 (default) | pi0_ur5e_peg_insertion_lora | peg_insertion_50demos/4000 | VLM=16, AE=32 | 4000 | Flow-matching |
| pi0.5 | pi05_ur5e_peg_insertion_lora | peg_insertion_50demos/4999 | VLM=16, AE=32 | 4999 | Flow-matching |
| pi0-FAST | pi0_fast_ur5e_peg_insertion_lora | peg_insertion_50demos/4999 | VLM=4 | 4999 | DCT broken |

### RL Token Checkpoints (in checkpoints/rl_token/)

| Model | File | embed | token | enc_layers | Loss | Step |
|-------|------|-------|-------|------------|------|------|
| pi0 (default) | pi0_ur5e_peg_insertion_lora_rl_token.pt | 2048 | 512 | 4 | 0.105 | 4971 |
| pi05 | pi05_ur5e_peg_insertion_lora_rl_token.pt | 2048 | 512 | 4 | - | - |
| pi0_fast | pi0_fast_ur5e_peg_insertion_lora_rl_token.pt | 2048 | 512 | 4 | - | - |
| Old (9-demo) | peg_insertion_9demos_v1.pt | 2048 | 512 | 2 | 0.038 | 4903 |

### Key Parameters (rlt/examples/peg_insertion/config.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| vla_config_name | pi0_ur5e_peg_insertion_lora | Flow-matching (no FAST) |
| rl_token_checkpoint | checkpoints/rl_token/pi0_ur5e_peg_insertion_lora_rl_token.pt | HPC-trained |
| max_residual_pos | 0.05 rad | ~2.9 deg per joint per step |
| chunk_size | 10 | Steps per RL decision |
| max_episode_chunks | 30 | 300 steps = 30s max |
| action_dim | 6 | Joint deltas (gripper removed) |
| token_dim | 512 | RL Token bottleneck |
| embed_dim | 2048 | Gemma-2B hidden size |
| warmup_episodes | 20 | VLA-only, no residual |
| utd_ratio | 5 | Updates per env step |
| batch_size | 256 | SAC batch size |
| use_classifier | False | Distance reward (10mm threshold) |

## 2. System Architecture

### Two-Terminal Setup

    Terminal 1 (VLA Server)              Terminal 2 (RLT Training)
    venv: openpi-ur5e/.venv              venv: ur5e_hil_serl/.venv
    Python: 3.11                         Python: 3.10
    JAX: GPU (serves VLA)                JAX: CPU (SAC training)
    Port: ws://0.0.0.0:8000             Connects to ws://localhost:8000

### Data Flow

    VLA Server (Terminal 1)
      - Loads pi0 checkpoint (5.8GB, rank=16)
      - Receives: images + state via WebSocket
      - Returns: 30 x 7 actions (joint deltas + gripper)
            |
            | WebSocket (ws://localhost:8000)
            v
    RLT Training (Terminal 2)
      1. VLAClient gets 30-step action chunk from VLA
      2. Take first 10 actions as reference (chunk_size=10)
      3. SAC agent outputs residual in [-1,1] x max_residual
      4. Combined: joint_delta = vla_ref + residual
      5. Jacobian: dx = J(q) * joint_delta -> Cartesian 6D
      6. Scale to SERL action space [-1,1]
      7. SERL env.step(cartesian_action) -> obs, reward
      8. Reward: distance < 10mm from TARGET_POSE -> reward=1

### Observation Space (after SERLObsWrapper + ChunkingWrapper)

    obs = {
        "state": shape (1, 19) float32       -- flattened proprio
        "wrist_1": shape (1, 128, 128, 3)    -- RealSense wrist
        "overview": shape (1, 128, 128, 3)   -- Kinect overhead
    }

    state[19] = flatten(Dict, ALPHABETICAL order):
      [0]     gripper_pose (1)     = 0.9686 constant (closed)
      [1:4]   tcp_force (3)        = force readings
      [4:10]  tcp_pose (6)         = RELATIVE xyz + euler
      [10:13] tcp_torque (3)       = torque readings
      [13:19] tcp_vel (6)          = velocity

### Action Spaces

    VLA output:     7D (6 joint deltas + 1 gripper) in radians
    SERL env input: 6D Cartesian (3 pos + 3 rot) in [-1, 1]
    Conversion:     J(q) * dq -> dx, then scale by ACTION_SCALE
    ACTION_SCALE:   [0.005m, 0.03rad, 1.0] (pos=5mm/unit, rot=0.03rad/unit)

### Safety

    Safety box (absolute TCP limits):
      X: [0.28, 0.42] m
      Y: [-0.02, 0.14] m
      Z: [0.03, 0.20] m
      Rot: +/-0.10 MRP = +/-22 deg
    
    Force mode limits: [0.5N, 0.5N, 0.5N, 1.0, 1.0, 1.0]
    Impedance controller provides compliance

## 3. Reward

### Distance-Based (Current)

    TARGET_POSE = [0.36066, 0.08130, 0.090, 2.200, -2.238, 0.006]
    REWARD_THRESHOLD = [0.010, 0.010, 0.010, 0.05, 0.05, 0.05]
    
    Reward = 1 when:
      - |TCP_xyz - TARGET_xyz| < 10mm (each axis)
      - rotation_error < 0.05 rad
    
    Verified: keyboard teleop reaches 1.1mm from target in ~43 steps

### Image Classifier (Disabled)

    The ResNet-10 binary classifier never exceeds prob=0.56 even at
    1mm from target. Needs retraining. Re-enable: use_classifier=True

## 4. Commands

### Terminal 1: VLA Server

    cd ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    .venv/bin/python scripts/serve_policy.py --port 8000 \
        policy:checkpoint \
        --policy.config pi0_ur5e_peg_insertion_lora \
        --policy.dir checkpoints/pi0_ur5e_peg_insertion_lora/peg_insertion_50demos/4000

### Terminal 2: RLT Training

    cd ~/ur5e_hande_workspace/rlt_ur5e
    source ur5e_hil_serl/.venv/bin/activate
    export JAX_PLATFORMS=cpu
    export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"
    python -m rlt.examples.peg_insertion.train_rlt

### Test Reward (no VLA needed)

    cd ~/ur5e_hande_workspace/rlt_ur5e
    source ur5e_hil_serl/.venv/bin/activate
    export JAX_PLATFORMS=cpu
    export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"
    python scripts/test_classifier.py              # distance reward
    python scripts/test_classifier.py --classifier # image classifier

## 5. Known Issues

1. pi0-FAST (rank=4) produces degenerate tokens (290, 360 repeating)
   -> Use pi0 (flow-matching) instead
2. Image classifier never fires (prob < 0.56 even at target)
   -> Use distance-based reward
3. State vector alphabetical ordering: gripper is at index 0, not pose
4. RelativeFrame transforms tcp_pose to be relative to reset pose
5. Base env distance reward terminates episodes even with classifier
   -> Fixed: classifier wrapper now overrides base env termination

## 6. Next Steps

1. Run full RLT training with pi0 VLA + distance reward
2. If pi0 actions are good, residuals should be small -> fast convergence
3. Try pi05 for potentially better VLA reference
4. Retrain classifier with more positive samples if needed
5. Add demo data to replay buffer (RLPD) for faster learning
