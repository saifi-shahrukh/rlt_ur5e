"""
RLT Configuration for Peg Insertion Task.

This config connects:
  - ur5e_hil_serl peg insertion environment (hardware, cameras, controller)
  - openpi VLA server (π0-FAST base policy)
  - RL Token model (state compression)
  - RLPD SAC agent (residual learning)

The same structure will be reused for PCB insertion and ethernet plugging
by changing only the task-specific parameters.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class RLTConfig:
    """All parameters for RLT training on peg insertion."""

    # ══════════════════════════════════════════════════════════════════════
    # Task Identity
    # ══════════════════════════════════════════════════════════════════════
    task_name: str = "peg_insertion"
    language_instruction: str = "Pick up the peg and insert it into the hole."

    # ══════════════════════════════════════════════════════════════════════
    # Hardware (copied from ur5e_hil_serl/experiments/peg_insertion/config.py)
    # ══════════════════════════════════════════════════════════════════════
    robot_ip: str = "172.22.1.139"
    control_hz: int = 10  # ur5e_hil_serl env control rate

    # Camera config (same as SERL peg insertion)
    realsense_cameras: dict = field(default_factory=lambda: {
        "wrist_1": {
            "serial_number": "034422070605",
            "dim": (640, 480),
            "exposure": 40000,
        },
    })
    kinect_cameras: dict = field(default_factory=lambda: {
        "overview": "000631452147",
    })

    # Image keys used by SERL wrappers
    image_keys: List[str] = field(default_factory=lambda: ["wrist_1", "overview"])
    classifier_keys: List[str] = field(default_factory=lambda: ["wrist_1", "overview"])

    # ══════════════════════════════════════════════════════════════════════
    # VLA Base Policy
    # ══════════════════════════════════════════════════════════════════════
    vla_config_name: str = "pi0_fast_ur5e_peg_insertion_lora"
    vla_checkpoint_dir: str = (
        "openpi_ur5e/openpi-ur5e/checkpoints/"
        "pi0_fast_ur5e_peg_insertion_lora/peg_insertion_9demos/29999"
    )
    vla_server_port: int = 8000  # OpenPI WebSocket server port
    vla_action_horizon: int = 30  # π0-FAST predicts 30 steps ahead

    # ══════════════════════════════════════════════════════════════════════
    # RL Token Model
    # ══════════════════════════════════════════════════════════════════════
    rl_token_checkpoint: str = "checkpoints/rl_token/peg_insertion_9demos_v1.pt"
    token_dim: int = 512       # RL token bottleneck dimension
    embed_dim: int = 2048      # VLM hidden size (Gemma-2B)

    # ══════════════════════════════════════════════════════════════════════
    # Observation / Action Dimensions
    # ══════════════════════════════════════════════════════════════════════
    # Proprio from SERL: tcp_pose(6, euler) + tcp_vel(6) + force(3) + torque(3) + gripper(1) = 19
    proprio_dim: int = 19
    # Action: 6D (GripperCloseEnv removes gripper dim for peg insertion)
    action_dim: int = 6
    # RL chunk size (how many steps the RL actor predicts at once)
    chunk_size: int = 10

    # ══════════════════════════════════════════════════════════════════════
    # RLPD / SAC Hyperparameters
    # ════════════════════════════════════���═════════════════════════════════
    # BC regularizer: L_actor = -Q + β * ||a - ã||²
    beta: float = 1.0          # Start at 1.0, decrease if too conservative
    # Reference action dropout (prevents policy from just copying VLA)
    ref_dropout: float = 0.5   # 50% of batches zero out reference
    # Update-to-data ratio (gradient steps per env step)
    utd_ratio: int = 5         # Lower than RLPD's 20 (chunks are correlated)
    # Critic ensemble (paper: TD3-style, 2 Q-functions, take min)
    critic_ensemble_size: int = 2
    critic_subsample_size: int = 2
    # SAC
    discount: float = 0.97
    soft_target_update_rate: float = 0.005
    backup_entropy: bool = False
    # Networks
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    # Optimizer
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4

    # ══════════════════════════════════════════════════════════════════════
    # Training Schedule
    # ══════════════════════════════════════════════════════════════════════
    # Warmup: VLA-only episodes to fill buffer before RL starts
    warmup_episodes: int = 20
    # Total episodes to train
    total_episodes: int = 800
    # When to start training (minimum transitions in online buffer)
    training_starts: int = 200
    # Batch size for RLPD updates
    batch_size: int = 256
    # Buffer capacity
    replay_buffer_capacity: int = 200_000
    # RLPD demo ratio (50% demo + 50% online)
    demo_ratio: float = 0.5

    # ══════════════════════════════════════════════════════════════════════
    # Logging / Checkpoints
    # ══════════════════════════════════════════════════════════════════════
    log_interval: int = 10      # Log every N episodes
    save_interval: int = 50     # Save checkpoint every N episodes
    checkpoint_dir: str = "checkpoints/rlt_runs/peg_insertion"
    use_wandb: bool = True
    wandb_project: str = "rlt-ur5e"

    # ══════════════════════════════════════════════════════════════════════
    # Reward
    # ══════════════════════════════════════════════════════════════════════
    # Use trained classifier (with consecutive frame filtering)
    use_classifier: bool = True
    classifier_checkpoint: str = "ur5e_hil_serl/examples/classifier_ckpt"
    classifier_threshold: float = 0.70
    consecutive_frames_needed: int = 3
    # Fallback: distance-based reward
    use_distance_reward: bool = False

    # ══════════════════════════════════════════════════════════════════════
    # SERL Demo Data (for RLPD demo buffer)
    # ══════════════════════════════════════════════════════════════════════
    demo_paths: List[str] = field(default_factory=lambda: [
        "ur5e_hil_serl/examples/demo_data/peg_insertion_194_transitions_2026-05-06_12-37-31.pkl",
        "ur5e_hil_serl/examples/demo_data/peg_insertion_608_transitions_2026-05-04_17-26-14.pkl",
        "ur5e_hil_serl/examples/demo_data/peg_insertion_937_transitions_2026-05-06_12-46-15.pkl",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # Safety
    # ══════════════════════════════════════════════════════════════════════
    # Max residual magnitude per step (clips RL output)
    max_residual_pos: float = 0.003   # 3mm max correction per step
    max_residual_rot: float = 0.02    # ~1° max rotation correction per step

    # ══════════════════════════════════════════════════════════════════════
    # agentlace Communication (same as ur5e_hil_serl)
    # ══════════════════════════════════════════════════════════════════════
    learner_ip: str = "localhost"
    learner_port: int = 5588
    learner_port_pub: int = 5589

    # ══════════════════════════════════════════════════════════════════════
    # Device
    # ══════════════════════════════════════════════════════════════════════
    device: str = "cuda"  # For VLA + RL Token (PyTorch)
    # JAX uses its own device detection


@dataclass
class PCBInsertionRLTConfig(RLTConfig):
    """PCB Insertion task — inherits from peg insertion, overrides task-specific params."""
    task_name: str = "pcb_insertion"
    language_instruction: str = "Insert the PCB into the slot."

    # Different checkpoint paths (will be created during fine-tuning)
    vla_config_name: str = "pi0_fast_ur5e_pcb_insertion_lora"  # TODO: create this config
    vla_checkpoint_dir: str = "openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_pcb_insertion_lora/run1/29999"
    rl_token_checkpoint: str = "checkpoints/rl_token/pcb_insertion_v1.pt"
    checkpoint_dir: str = "checkpoints/rlt_runs/pcb_insertion"

    # Different demo data
    demo_paths: List[str] = field(default_factory=lambda: [])
    # TODO: collect PCB insertion demos


@dataclass
class EthernetInsertionRLTConfig(RLTConfig):
    """Ethernet cable insertion — per the RLT paper's target task."""
    task_name: str = "ethernet_insertion"
    language_instruction: str = "Insert the ethernet cable into the port."

    # Different checkpoint paths
    vla_config_name: str = "pi0_fast_ur5e_ethernet_insertion_lora"  # TODO: create
    vla_checkpoint_dir: str = "openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_ethernet_lora/run1/29999"
    rl_token_checkpoint: str = "checkpoints/rl_token/ethernet_insertion_v1.pt"
    checkpoint_dir: str = "checkpoints/rlt_runs/ethernet_insertion"

    # Different demo data
    demo_paths: List[str] = field(default_factory=lambda: [])
    # TODO: collect ethernet demos

    # Ethernet may need tighter safety (thinner cable)
    max_residual_pos: float = 0.002   # 2mm
    max_residual_rot: float = 0.015   # ~0.8°


# ══════════════════════════════════════════════════════════════════════════
# Config Registry (like ur5e_hil_serl's CONFIG_MAPPING)
# ══════════════════════════════════════════════════════════════════════════

RLT_CONFIG_MAPPING = {
    "peg_insertion": RLTConfig,
    "pcb_insertion": PCBInsertionRLTConfig,
    "ethernet_insertion": EthernetInsertionRLTConfig,
}
