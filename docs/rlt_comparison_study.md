# RLT Comparison Study: vla-rlt vs RLinf/piRL vs Our rlt_ur5e

## Executive Summary

Three approaches to online RL fine-tuning of VLAs exist:
1. **Pi's RLT** (Physical Intelligence): RL Token as information bottleneck, SAC + residual
2. **piRL** (RLinf): Full policy gradient (PPO/GRPO) on the flow-matching model itself
3. **vla-rlt** (akashspacesky): Open-source RLT reimplementation with SmolVLA/GR00T
4. **Our rlt_ur5e**: Attempted RLT with pi0 + external SAC

---

## 1. Architecture Comparison

### 1.1 vla-rlt (akashspacesky/vla-rlt)

**Approach**: Additive delta on VLA actions via SAC actor-critic

    Images + Language
          |
    [VLA (GR00T/SmolVLA)] <-- FROZEN
          |
    hidden_states (from multiple VLM layers via hooks)
          |
    [RLT Encoder] (transformer + cross-attention pooling)
          |
    rl_token (d=128 or 256)
          |
    [SAC Actor MLP] + vla_action --> delta_action
          |
    final_action = vla_action + delta

**Key Design Decisions:**
- VLA is FROZEN entirely (no fine-tuning during RL)
- Hidden states extracted via HOOKS inside the VLA inference
- RLT Encoder uses cross-attention pooling (query token attends to sequence)
- Actor takes BOTH rl_token AND vla_action as input
- Output is ADDITIVE DELTA (not replacement)
- Reference regularization: L2 penalty ||delta||^2 toward zero
- Two-phase training:
  - Phase 1: Offline bottleneck pre-training (MSE reconstruction)
  - Phase 2: Online SAC with frozen encoder

**SAC Details:**
- State: rl_token (d=128/256)
- Action: delta_action_chunk (chunk_size * action_dim)
- Critic input: (rl_token, full_action_chunk)
- Actor loss: SAC entropy + ref_reg_weight * ||delta - 0||^2
- Auto-entropy tuning (alpha)
- Soft target update (tau=0.005)
- updates_per_step = 50 (many gradient steps per env step)

### 1.2 piRL (RLinf/RLinf - arXiv:2510.25889)

**Approach**: Direct policy gradient on the flow-matching model

    Images + Language
          |
    [pi0 / pi0.5 VLA] <-- TRAINED via RL
          |
    flow-matching denoising (T steps)
          |
    actions (continuous)

**Key Innovation:**
The challenge: flow-matching models don't have tractable log-likelihoods
(needed for policy gradient). piRL proposes two solutions:

1. **Flow-Noise**: Models denoising as discrete-time MDP with learnable
   noise network for exact log-likelihood computation.

2. **Flow-SDE**: Integrates denoising with environment interaction,
   formulating a two-layer MDP. Uses ODE-to-SDE conversion.

**Key Design Decisions:**
- VLA is NOT frozen — the entire model is fine-tuned via RL
- Uses PPO/GRPO (not SAC)
- No separate RL token or actor-critic
- Ray-based distributed training (actor/learner/env workers)
- FSDP for model parallelism
- Works on sim benchmarks (LIBERO, ManiSkill3, MetaWorld, CALVIN)

**Results:**
- pi0 on ManiSkill3: 38.4% (SFT) -> 78.8% (Flow-SDE)
- pi0.5 on ManiSkill3: 40.1% (SFT) -> 90.9% (Flow-SDE)
- pi0.5 on CALVIN ABC-D: 61.3% (SFT) -> 87.0% (Flow-SDE)

### 1.3 Our rlt_ur5e

**Approach**: External SAC with Jacobian conversion

    Images + Language
          |
    [pi0 VLA] <-- FROZEN, WebSocket server
          |
    absolute joint targets (30 steps)
          |
    delta_q = target - current (joint deltas)
          |
    [Jacobian J(q)] --> Cartesian deltas
          |
    [SAC Agent] outputs residual (joint space)
          |
    combined = vla_cartesian + sac_residual_cartesian
          |
    SERL env.step(cartesian_action)

---

## 2. Critical Differences

| Aspect | vla-rlt | piRL | Our rlt_ur5e |
|--------|---------|------|---------------|
| VLA modified? | No (frozen) | Yes (full RL) | No (frozen) |
| RL algorithm | SAC | PPO/GRPO | SAC |
| Action representation | Additive delta | Full action | Additive delta |
| RL state input | rl_token (from hooks) | Full VLA state | proprio + images |
| Where RL token? | Inside VLA process | N/A (no token) | Separate model |
| Action space | Same as VLA (joints) | Same as VLA | Different (Cartesian) |
| VLA access | Full (hooks, hidden states) | Full (gradients) | WebSocket only |
| Training speed | ~15 min real robot | Hours (sim) | Not converging |
| Deployment | Single process | Distributed | Two processes |
| Demo usage | Replay buffer (RLPD) | SFT pretraining | None (missing!) |

## 3. Root Cause Analysis: Why Our System Doesn't Work

### 3.1 The Fundamental Problem

**vla-rlt extracts hidden states from INSIDE the VLA via hooks.**
**We access the VLA only via WebSocket (actions out, no internals).**

This means:
- We CANNOT extract rl_token during online RL (no hidden state access)
- Our RL Token model was pre-trained on offline embeddings, but at inference
  it receives ZERO (since we can't call the VLA hooks from a separate process)
- The SAC agent receives (proprio + image features) instead of rl_token
- This is fundamentally different from the RLT paper's design

### 3.2 Why vla-rlt Works and Ours Doesn't

| Design Element | vla-rlt (works) | Our rlt_ur5e (broken) |
|----------------|-----------------|------------------------|
| VLA runs in | Same process as RL | Separate WebSocket server |
| Hidden states | Extracted via hooks | Not accessible |
| RL Token at inference | Fresh from current obs | Always zero |
| Actor input | rl_token + vla_action | proprio (591D) |
| Action space | Same as VLA (joints) | Different (Cartesian) |
| Reference action | VLA action (direct) | VLA action (through Jacobian) |
| Demo buffer | Yes (RLPD) | No |
| Updates per step | 50 | 5 |

### 3.3 Why piRL is Not Applicable to Us

- piRL requires backpropagating through the VLA (all parameters)
- Needs GPU-hours of training in simulation
- Our VLA is frozen and accessed via WebSocket
- We're doing real-robot training (no sim)
- piRL's approach is for large-scale sim-to-real, not our use case

---

## 4. What We Should Actually Build

### 4.1 Option A: Proper RLT (Single-Process, With Hooks)

Match vla-rlt's architecture:

    [pi0 VLA + hooks] ← runs in-process (needs JAX + 6GB GPU)
         |
    hidden_states (fresh, every step)
         |
    [RLT Encoder] → rl_token (512D)
         |
    [SAC Actor(rl_token, vla_action)] → delta_action
         |
    final_joints = vla_joints + delta
         |
    robot.servoJ(final_joints)

**Requires:** VLA and RL in same process, or shared-memory GPU access
**Pros:** Matches paper exactly, proven to work
**Cons:** Need to run VLA + SAC on same GPU, or use MPS/shared memory

### 4.2 Option B: Simplified Residual (No RL Token, Just SAC + VLA)

Drop the RL Token entirely. Use SAC with observations directly:

    [pi0 VLA server] → vla_joint_targets (via WebSocket)
         |
    [SAC Agent(proprio, vla_action)] → residual_joints
         |
    final_joints = vla_joints + residual
         |
    robot.servoJ(final_joints)

**Key:** SAC observes (joint_pos, joint_vel, force, vla_action) = ~25D
**Requires:** Demo buffer (RLPD) for sparse reward
**Pros:** Simple, works with WebSocket VLA, no hooks needed
**Cons:** Loses information bottleneck benefit of RL Token

### 4.3 Option C: Current Setup Fixed (Minimum Changes)

Keep current architecture but fix critical issues:

1. ADD demo buffer loading (RLPD) ← CRITICAL
2. Increase max_residual to 0.1 rad
3. Use VLA reference as SAC input (concat with obs)
4. Add reference regularization loss (||residual||^2)
5. Increase updates_per_step to 20-50

This is essentially Option B but keeping the Jacobian conversion
(suboptimal but functional with enough demos).

---

## 5. Comparison with Pi's Original RLT Paper

### 5.1 What the Paper Actually Does

From the Pi website and paper:
- RL Token is extracted from pi0.5's internal representations
- A small MLP actor-critic (not the VLA itself) is trained with SAC
- The actor outputs a RESIDUAL on top of VLA's base action
- Key insight: the RL token provides sufficient state representation
  for a tiny network to learn precise corrections
- Training: ~15 minutes of real robot data
- Result: 2.7x improvement on ethernet insertion (147→400 per 10min)

### 5.2 What Makes It Work

1. **Information bottleneck**: RL token compresses VLA state to ~128-512D
2. **Additive residual**: Never replaces VLA, only refines
3. **Reference regularization**: Prevents RL from diverging from VLA
4. **RLPD**: 50% demo sampling from successful trajectories
5. **Same action space**: RL operates in same space as VLA (joints)
6. **Fast updates**: 50 gradient steps per environment step
7. **Small networks**: Actor/critic are tiny MLPs (~3 layers, 256 hidden)

### 5.3 What We Got Wrong

| Pi's Design | Our Implementation | Gap |
|-------------|-------------------|-----|
| RL token from hooks | RL token is zero (no hooks) | Critical |
| Same action space | Mismatch (joints→Cartesian) | Major |
| RLPD demos | No demo buffer | Critical |
| ref_reg ||delta||^2 | No reference reg | Moderate |
| 50 updates/step | 5 updates/step | Major |
| servoJ (direct) | Impedance (indirect) | Moderate |
| VLA in-process | VLA via WebSocket | Architectural |

---

## 6. Recommended Path Forward

### Immediate (Today → This Week)

1. **Add RLPD demo loading** — Convert SERL demo PKL files to replay buffer
2. **Increase UTD to 20** — More gradient steps per env step
3. **Add reference regularization** — ||residual||^2 loss term
4. **Increase max_residual to 0.1 rad** — More exploration room
5. **Concat VLA reference to SAC obs** — SAC sees what VLA recommends

### Short-term (This Week → Next Week)

6. **Add servoJ mode** — Direct joint control, bypass Jacobian
7. **VLA + residual in joint space** — Eliminate Cartesian conversion
8. **Complete VLA training (30k steps)** — Meaningful VLA reference

### Medium-term (Next 2-4 Weeks)

9. **Run VLA in-process** — Enable hidden state extraction
10. **Use RL Token properly** — Fresh embeddings each step
11. **Match vla-rlt architecture** — Full RLT with hooks

### End Goal

    [pi0 VLA in-process (JAX, GPU)]
         |
    hooks → hidden_states
         |
    [RLT Encoder (frozen)] → rl_token (512D)
         |
    [SAC Actor(rl_token, vla_action)] → delta_joints
         |
    final_joints = vla_action + delta
         |
    robot.servoJ(final_joints, gain=300)
         |
    reward = distance_to_target < 10mm
         |
    [SAC update: 50 steps/env_step, with RLPD]

---

## 7. Key Takeaways

1. **The RL Token is only useful if you can extract it LIVE** during RL training.
   Without hooks into the VLA, there's no RL token — just a pre-trained
   embedding that's stale/zero at inference.

2. **RLPD (demo buffer) is NON-NEGOTIABLE** for sparse reward tasks.
   Without it, SAC cannot discover the reward signal through random exploration.

3. **Action space consistency is critical.** VLA outputs joints → RL should
   also operate in joints → robot should accept joints (servoJ).

4. **The WebSocket separation is architecturally incompatible with RLT.**
   For proper RLT, VLA must run in the same process as RL.
   For a simplified residual approach, WebSocket is fine but need RLPD.

5. **piRL (full VLA fine-tuning) is a different paradigm** — requires sim,
   GPU-hours, and policy gradient through flow-matching. Not applicable
   to our real-robot, few-episode setup.
