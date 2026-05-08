# HIL-SERL: Human-in-the-Loop Sample-Efficient RL

## Purpose
Provides the **robot control layer** and **reward classifier** for the UR5e peg insertion task.

## Architecture
```
ur5e_hil_serl/
├── serl_robot_infra/          ← Low-level robot control (RTDE + Robotiq gripper)
│   └── ur_env/
│       └── envs/
│           ├── ur5e_env.py    ← Base UR5e Gymnasium environment
│           ├── relative_env.py← Relative frame control (TCP delta actions)
│           └── wrappers.py    ← GripperClose, KeyboardIntervention, Classifier
├── serl_launcher/             ← RL training infrastructure
│   ├── agents/continuous/sac.py  ← SAC with image encoders
│   └── wrappers/
│       ├── serl_obs_wrappers.py  ← State extraction (proprio keys)
│       └── chunking.py           ← Observation stacking
└── examples/
    ├── experiments/peg_insertion/
    │   ├── config.py          ← EnvConfig + TrainConfig
    │   └── wrapper.py         ← PegInsertionEnv (task-specific reset)
    ├── classifier_ckpt/       ← Trained image reward classifier
    ├── classifier_data/       ← Success/failure images for training
    ├── demo_data/             ← Teleoperated demonstrations (.pkl)
    └── train_rlpd.py          ← Original actor-learner training script
```

## Key Components Used by RLT

| Component | How RLT Uses It |
|-----------|----------------|
| `PegInsertionEnv` | Wraps the UR5e hardware (RTDE + gripper + cameras) |
| `MultiCameraBinaryRewardClassifierWrapper` | Provides sparse +1 reward when peg is inserted |
| `GripperCloseEnv` | Keeps gripper closed (peg pre-grasped) |
| `SERLObsWrapper` | Extracts proprio: tcp_pose, tcp_vel, force, torque, gripper |
| `demo_data/*.pkl` | Pre-collected demonstrations for RLPD buffer |

## Reward Classifier
- **Input:** Wrist + overview camera images
- **Output:** Binary success probability
- **Trigger:** 3 consecutive frames > 0.70 threshold → reward=1
- **Location:** `examples/classifier_ckpt/checkpoint_150/`

## Venv
- **Path:** `ur5e_hil_serl/.venv/`
- **Python:** 3.10
- **Key packages:** JAX, Flax, PyTorch, ur_rtde, gymnasium
