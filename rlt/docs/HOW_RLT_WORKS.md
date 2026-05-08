# How RLT Works — The Complete Pipeline Explained

## Your Confusion Answered

> "First I need 50 demos to fine-tune, then directly train on hardware with actor-critic?"

**YES, exactly.** RLT has two distinct phases:

---

## Phase A: OFFLINE (No RL, just preparation)

```
┌────────────────────────────────────────────────────────────┐
│  STEP 1: Collect 50 demonstrations (keyboard teleoperation)│
│           → Same recording you already did for openpi      │
│           → LeRobot format (images + actions at 30fps)     │
│           → You already have 4, need ~50 total             │
│                                                            │
│  STEP 2: Fine-tune VLA (π0-FAST / π0 / π0.5)             │
│           → You ALREADY DID THIS (checkpoint exists!)      │
│           → pi0_fast_ur5e_peg_insertion_lora/29999         │
│           → This gives base policy: ~20-40% success rate   │
│                                                            │
│  STEP 3: Train RL Token encoder (30 min, offline)          │
│           → Run VLA on demo data, extract embeddings       │
│           → Train encoder-decoder to compress them          │
│           → Output: rl_token checkpoint (10MB)             │
└────────────────────────────────────────────────────────────┘
                         │
                         │ All three outputs ready:
                         │   ✓ VLA checkpoint (base policy)
                         │   ✓ RL Token checkpoint (state encoder)
                         │   ✓ SERL demo transitions (for demo buffer)
                         ▼
```

## Phase B: ONLINE (Real robot RL — the actual RLT training)

```
┌────────────────────────────────────────────────────────────┐
│  STEP 4: Run actor-critic on real robot (3-10 days)        │
│                                                            │
│  Every episode (~10 seconds):                              │
│                                                            │
│    1. Robot starts at reset position (peg above hole)      │
│                                                            │
│    2. VLA server predicts action chunk ã (10 steps)        │
│       → This is the "base behavior" — rough insertion      │
│                                                            │
│    3. RL actor predicts RESIDUAL (small correction)        │
│       → Input: z_rl + proprio + reference ã                │
│       → Output: Δa (tiny adjustments, <2mm per step)       │
│                                                            │
│    4. Execute: final_action = ã + Δa on robot              │
│                                                            │
│    5. Reward: classifier says success/fail                  │
│                                                            │
│    6. Store transition in replay buffer                     │
│                                                            │
│    7. Learner does 5 gradient updates (fast!)              │
│                                                            │
│  Human can intervene with keyboard if robot gets stuck     │
│  → Corrections go straight into demo buffer                │
│                                                            │
│  Result: 20-40% → 80-95% success in ~800 episodes         │
└────────────────────────────────────────────────────────────┘
```

---

## For YOUR Peg Insertion Task Specifically

### What you ALREADY have:
- ✅ 4 LeRobot demos (need ~46 more)
- ✅ π0-FAST checkpoint trained (30k steps)
- ✅ 1739 SERL demo transitions (for RL demo buffer)
- ✅ Reward classifier trained (200 success + 2660 failure images)
- ✅ Consecutive-frame filtering (3 frames >0.70 for reward)
- ✅ Working impedance controller + env + wrappers
- ✅ RL Token model code (tested, 23/23 tests pass)

### What you STILL need:
- 🔲 ~46 more LeRobot demos (for better VLA if needed)
- 🔲 RL Token trained on real VLA embeddings
- 🔲 UR5eRLTEnv wrapper (connects VLA → RL Token → SAC)
- 🔲 RLT actor/learner scripts
- 🔲 Online training (800 episodes on robot)

### BUT — here's the shortcut:

Since you ALREADY have a trained π0-FAST checkpoint AND 1739 SERL demos,
you can start Phase B (online RL) with what you have:

1. Use existing π0-FAST as base policy (even with 4 demos, it moves roughly right)
2. Use existing SERL demos for the demo buffer
3. Use existing reward classifier for reward signal
4. The RL will learn the residual corrections on top of whatever the VLA does

Collecting more demos would improve the VLA base (making RL's job easier),
but it's not strictly required to START.

---

## The Three Tasks

| Task | Status | What's Needed |
|------|--------|---------------|
| **Peg Insertion** | Ready to start | RL Token + env wrapper |
| **PCB Insertion** | Config exists | + demos + VLA fine-tune |
| **Ethernet Plug** | New task | + demos + VLA fine-tune + reward classifier |

All three use the SAME RLT infrastructure — only the task config changes:
- Different reset position (RESET_Q)
- Different target pose (TARGET_POSE)
- Different safety box (ABS_POSE_LIMIT)
- Different reward (classifier or distance-based)

---

## Timing Estimate

| Step | Time | Can Skip? |
|------|------|----------|
| Collect 50 more demos | 2 hours | Yes (start with 4) |
| Re-train VLA on 50 demos | 4 hours (GPU) | Yes (use existing) |
| Train RL Token | 30 min | No — needed |
| Online RL training | 3-10 days | No — this IS the goal |
| **Total to first results** | **~1 day** (if using existing VLA) | |
