# System Comparison & Architecture Analysis

## PART 1 — HIL-SERL

### 1.1 Architecture (rail-berkeley/hil-serl)

    ACTOR NODE (CPU)                         LEARNER NODE (GPU)
    ================                         ==================
    - Robot gym env                          - SAC agent (train)
    - Camera capture                         - Replay buffer
    - Policy inference                       - Demo buffer (RLPD)
    - SpaceMouse intervention                - WandB logging
    - Data collection                        - Checkpoint saving
         |                                        |
         +---- agentlace (network) ---------------+
              (transitions → learner)
              (params ← learner)

**Key Components:**
- SAC with RLPD (50% demo sampling from replay buffer)
- Image-based observations (ResNet encoder, pretrained)
- SpaceMouse/keyboard human intervention during training
- Intervention data stored separately (richer signal)
- Binary reward classifier (learned from human labels)
- Asynchronous actor-learner via agentlace TCP

**Training Loop (Actor Side):**
1. Sample action from SAC policy
2. env.step(action)
3. If human intervenes via SpaceMouse → override action, mark transition
4. Store transition in data_store (and intvn_data_store if intervened)
5. Send stats to learner
6. Receive updated params from learner

**Training Loop (Learner Side):**
1. Sample batch: 50% from online buffer, 50% from demo buffer
2. SAC update (critic + actor + alpha)
3. Periodically sync params to actor
4. Log metrics, save checkpoints

### 1.2 ur5e_hil_serl (Our Adaptation)

**Preserved:**
- SAC agent (serl_launcher/agents/continuous/sac.py)
- Replay buffer structure
- Reward classifier
- Wrapper stack (GripperClose, RelativeFrame, Quat2Euler, SERLObs, Chunking)
- Keyboard/SpaceMouse intervention

**Modified:**
- Replaced Franka robot infra with UR5e RTDE controller
- Added Kinect V2 camera support
- Impedance control via UR5e forceMode (not Franka's built-in)
- Simplified to single-process (no agentlace, no actor-learner split)
- Added keyboard teleop (FakeSpaceMouseExpert via pynput)

**Removed:**
- ROS dependency (direct RTDE instead)
- Flask robot server (direct controller process)
- Dual-arm support

### 1.3 Comparison Table

| Aspect | HIL-SERL (Original) | ur5e_hil_serl |
|--------|---------------------|---------------|
| Robot | Franka Panda | UR5e + Hand-E |
| Communication | ROS + Flask | Direct RTDE |
| Actor/Learner | Separate (agentlace) | Single process |
| RL Algorithm | SAC + RLPD | SAC + RLPD |
| Intervention | SpaceMouse | Keyboard (pynput) |
| Reward | Classifier + distance | Classifier + distance |
| Control | Franka impedance | forceMode impedance |
| Cameras | RealSense | RealSense + Kinect |
| Encoder | ResNet-10 pretrained | ResNet-10 pretrained |

---

## PART 2 — OPENPI

### 2.1 Core OpenPI Architecture

    DATA (LeRobot/HuggingFace)
         |
         v
    TRAINING (scripts/train.py)
    - Pi0Config / Pi0FASTConfig
    - PaLI-Gemma 2B backbone (frozen or LoRA)
    - Flow-matching head (continuous) OR FAST tokenizer (discrete)
    - Action horizon: 30-50 steps predicted at once
    - Norm stats: quantile normalization per dataset
         |
         v
    CHECKPOINT (orbax, params/)
         |
         v
    SERVING (scripts/serve_policy.py)
    - WebSocket server (port 8000)
    - Receives: obs dict (images + state + prompt)
    - Returns: actions array (absolute joint targets)
    - Auto-resizes images to 224x224 internally
         |
         v
    CLIENT (openpi_client)
    - WebsocketClientPolicy.infer(obs) -> {"actions": ndarray}

**Model Types:**
- pi0: Flow-matching (continuous actions, diffusion-like denoising)
- pi0-FAST: Autoregressive tokens (DCT-based action compression)
- pi0.5: Enhanced pi0 with knowledge insulation

### 2.2 F-Fer/openpi-ur5e (Fork for UR5e)

**Changes from upstream OpenPI:**
- Added UR5e-specific data config (LeRobotUR5DualCamDataConfig)
- Added ur5e_policy.py (UR5EInputs, UR5EOutputs transforms)
- Camera mapping: overhead→exterior_image_1_left, wrist→wrist_image_left/right
- Norm stats for UR5e joint ranges
- Training configs for LoRA fine-tuning (rank 4/16)
- 224x224 image target size (PaLI-Gemma standard)

**Key Design:**
- Training: HuggingFace datasets → fine-tune LoRA → checkpoint
- Inference: serve_policy.py WebSocket → client queries
- Actions: Absolute joint positions (7D: 6 joints + gripper)
- Normalization: quantile (q01/q99) per action dimension

### 2.3 Comparison

| Aspect | OpenPI (upstream) | openpi-ur5e (F-Fer) | Our openpi_ur5e |
|--------|-------------------|---------------------|------------------|
| Robot | DROID/ALOHA/etc | UR5e | UR5e |
| Images | 224x224 | 224x224 | 128 (auto-resized) |
| Actions | Absolute joints | Absolute joints | Absolute joints |
| Control | N/A (inference) | servoJ via LeRobot | Impedance via SERL |
| Fine-tuning | Yes (LoRA) | Yes (LoRA) | Yes (LoRA) |
| Deployment | WebSocket | WebSocket + LeRobot | WebSocket + RLT |

---

## PART 3 — LEROBOT_UR5E_GELLO

### 3.1 Purpose

This is a **data collection and VLA deployment** system. NOT RL training.

    GELLO (leader arm)                  UR5e (follower arm)
    ==================                  ==================
    - 7-DOF Dynamixel                   - RTDE joint control
    - Human teleoperates                - Follows GELLO commands
    - Records joint angles              - Records camera images
         |                                   |
         +------- LeRobot Dataset -----------+
                  (HuggingFace format)
                       |
                       v
              OpenPI Fine-tuning
                       |
                       v
              VLA Policy Inference
              (remote_pi_inference_dual_cam.py)
                       |
                       v
              UR5e Autonomous Execution
              (servoJ with action chunks)

**Components:**
- lerobot_robot_ur5e: UR5e RTDE plugin for LeRobot
- lerobot_teleoperator_gello: GELLO Dynamixel plugin
- lerobot_camera_zmq: Raspberry Pi camera streaming
- scripts/remote_pi_inference_dual_cam.py: VLA deployment

**Key Design:**
- Action space: Joint positions (absolute, radians)
- Control: servoJ (direct joint position commands)
- Inference: Action chunking (predict 30, execute all sequentially)
- NO RL component — pure imitation learning deployment

### 3.2 How it Differs

| Aspect | lerobot_ur5e_gello | HIL-SERL | rlt_ur5e |
|--------|-------------------|----------|----------|
| Purpose | Data collection + VLA deploy | Online RL training | VLA + online RL |
| Learning | Imitation only | RL (SAC) | RL (SAC) + VLA ref |
| Control | servoJ (joint) | Impedance (Cartesian) | Impedance (Cartesian) |
| Human role | Teleoperation | Intervention during RL | None (automated) |
| Action space | Absolute joints | Cartesian delta | Joint delta→Cartesian |

---

## PART 4 — SYSTEM-LEVEL SYNTHESIS

### 4.1 Conceptual Stack

    Layer 5: TASK EXECUTION
    ├── HIL-SERL:        SAC policy → Cartesian actions → impedance
    ├── lerobot_gello:   VLA policy → joint actions → servoJ
    └── rlt_ur5e:        VLA + SAC residual → joint delta → Jacobian → Cartesian

    Layer 4: POLICY
    ├── HIL-SERL:        SAC (MLP + ResNet encoder, trained online)
    ├── OpenPI:          VLA (PaLI-Gemma + flow/FAST, trained offline)
    └── rlt_ur5e:        VLA reference + SAC residual (hybrid)

    Layer 3: TRAINING
    ├── HIL-SERL:        Online RL with human corrections (RLPD)
    ├── OpenPI:          Offline supervised (behavior cloning on demos)
    └── rlt_ur5e:        Offline VLA + Online RL refinement

    Layer 2: DATA
    ├── HIL-SERL:        Replay buffer (online) + demo buffer (offline)
    ├── lerobot_gello:   HuggingFace dataset (teleoperated demos)
    └── rlt_ur5e:        Replay buffer + VLA embeddings

    Layer 1: HARDWARE
    ├── HIL-SERL:        Franka + ROS + impedance controller
    ├── lerobot_gello:   UR5e + RTDE + servoJ
    └── rlt_ur5e:        UR5e + RTDE + forceMode impedance

### 4.2 Where Systems Overlap

- HIL-SERL and rlt_ur5e: Both use SAC, image observations, impedance control
- OpenPI and lerobot_gello: Both use VLA, absolute joint actions, servoJ
- rlt_ur5e attempts to BRIDGE these two paradigms

### 4.3 Where They Diverge

- **Action space**: HIL-SERL uses Cartesian deltas. OpenPI/lerobot use absolute joints.
  rlt_ur5e awkwardly converts between them.
- **Control**: HIL-SERL uses impedance (force-compliant). OpenPI uses servoJ (stiff).
  rlt_ur5e uses impedance but receives joint commands.
- **Learning**: HIL-SERL is pure online RL. OpenPI is pure offline IL.
  rlt_ur5e is a hybrid that hasn't fully committed to either paradigm.

### 4.4 What's Missing in a Unified Architecture

1. **Consistent action space**: Pick one (joints or Cartesian) end-to-end
2. **Demo loading for RLPD**: The single biggest missing piece for convergence
3. **Direct joint control option**: servoJ mode for VLA-only evaluation
4. **VLA quality metrics**: No way to measure VLA-only success rate cleanly

---

## PART 5 — EVALUATION OF rlt_ur5e

### 5.1 What is Correctly Designed

| Component | Assessment |
|-----------|------------|
| Two-terminal architecture | Correct. Separates GPU-heavy VLA from CPU SAC. |
| WebSocket VLA client | Correct. Standard OpenPI interface. |
| Distance-based reward | Correct. Proven working (1.1mm accuracy). |
| Safety box + impedance | Correct. Prevents damage during exploration. |
| Chunk-based execution | Correct. Matches VLA action horizon concept. |
| RL Token model | Correct architecture (encoder-decoder). Needs better training. |
| JIT SAC agent | Correct. Fast inference (0.6ms). |

### 5.2 What is Incorrect or Inconsistent

| Issue | Problem | Impact |
|-------|---------|--------|
| Action space mismatch | VLA outputs joints, SERL expects Cartesian | Extra Jacobian layer adds errors |
| No demo buffer loading | SAC has no signal in sparse reward | Cannot converge without demos |
| VLA undertrained | 4000/30000 steps = identity mapping | VLA reference is useless noise |
| Hardcoded zeros (now fixed) | Was returning zeros instead of VLA actions | Robot didn't move |
| Image classifier broken | Never fires even at target | Disabled, but needs retraining |
| max_residual too small | 0.05 rad when VLA gives ~0.01 rad deltas | SAC corrections limited |

### 5.3 What is Over-Engineered

| Component | Why Over-Engineered |
|-----------|--------------------|
| Jacobian conversion | servoJ would eliminate this entirely |
| RL Token (for now) | Without VLA embeddings from inference, z_rl is always zero |
| 3-camera duplication | Training data has 2 cameras; duplicating wrist wastes compute |
| Chunk size = 10 | With VLA predicting 30 steps, could use all 30 directly |

### 5.4 What is Missing

| Missing Component | Why Critical |
|------------------|--------------|
| RLPD demo loading | SAC cannot discover sparse reward without demos |
| VLA-only eval mode | Need to measure VLA quality independently |
| Action logging/plotting | Cannot debug what's happening without visualization |
| Checkpoint resume | Training crashes lose all progress |
| servoJ option | Direct joint control for VLA evaluation |
| Success rate tracking | No persistent success rate across sessions |

### 5.5 Comparison Against Design Principles

**vs HIL-SERL principles:**
- Missing: RLPD (50% demo sampling). This is THE key ingredient.
- Missing: Human intervention during RL training.
- Present: SAC, image observations, reward function.
- Deviation: Single process instead of actor-learner split (acceptable for now).

**vs OpenPI VLA paradigm:**
- Present: WebSocket serving, standard obs format.
- Problem: VLA actions converted through Jacobian instead of used directly.
- Problem: 128x128 images (auto-resized, but original demo data was different pipeline).
- Missing: Direct servoJ deployment for VLA-only baseline.

**vs LeRobot approach:**
- Missing: Clean VLA-only execution loop (action chunk → servoJ → repeat).
- Missing: Dataset tooling integration for RLPD.
- Present: Camera infrastructure, RTDE control.

### 5.6 Recommended Target Architecture

    VLA Server (OpenPI, GPU)         RL Training (CPU/small GPU)
    ========================         ==========================
    pi0 checkpoint (30k steps)       SAC agent
    WebSocket :8000                  Replay buffer + Demo buffer
    Returns: absolute joints              |
         |                                |
         v                                v
    UNIFIED CONTROL LOOP (single process)
    ====================================
    1. obs = robot.get_observation()          [RTDE + cameras]
    2. vla_actions = vla_client.infer(obs)    [absolute joints]
    3. residual = sac.sample(state)           [joint deltas]
    4. target_q = vla_actions[0] + residual   [absolute joints]
    5. robot.servoJ(target_q, gain=300)       [direct joint control]
    6. reward = check_distance(tcp, target)   [distance reward]
    7. buffer.insert(transition)
    8. sac.update(buffer.sample())

**Key changes from current:**
- Use servoJ instead of Cartesian impedance (no Jacobian needed)
- VLA + residual in JOINT SPACE (same space, additive)
- Load demos into replay buffer (RLPD)
- Action format: absolute joint targets (not deltas converted to Cartesian)

### 5.7 Concrete Next Steps (Priority Order)

1. **Load demo transitions into RLPD buffer** — HIGHEST PRIORITY
   Without this, SAC cannot learn from sparse reward.

2. **Complete VLA training to 30k steps** (HPC)
   Current 4000 steps produces identity mapping.

3. **Add servoJ control mode** (bypass Jacobian)
   VLA outputs joints → add residual → servoJ.
   Eliminates the Jacobian conversion error.

4. **Increase max_residual to 0.1 rad**
   With VLA giving tiny actions, SAC needs room to explore.

5. **Run overnight with demo buffer loaded**
   Should converge in 200-400 episodes with RLPD.

### 5.8 Minimal Path to Working System (Today)

Given: VLA at 4000 steps (weak), current Jacobian pipeline:

1. Load SERL demo data into replay buffer (50% sampling ratio)
2. Increase max_residual_pos to 0.1 rad
3. Run RLT training with distance reward
4. Expected: convergence in 200-500 episodes
5. Measure: SR(last 50) should reach >50% within 2-3 hours

This works because RLPD provides the exploration signal that pure
random exploration cannot find in sparse reward tasks.
