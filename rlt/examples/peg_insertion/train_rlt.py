#!/usr/bin/env python3
"""
RLT Training Script for Peg Insertion.

Two modes of operation:
  1. VLA-Only (baseline): Just execute VLA actions, measure success rate
  2. VLA + RLT (full): VLA provides reference, SAC learns residual corrections

Usage:
  cd ~/ur5e_hande_workspace/rlt_ur5e
  source ur5e_hil_serl/.venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"

  # VLA-only baseline (no RL):
  python -m rlt.examples.peg_insertion.train_rlt --warmup_only

  # Full RLT training (VLA + RL Token + SAC):
  python -m rlt.examples.peg_insertion.train_rlt

  # Evaluate saved checkpoint:
  python -m rlt.examples.peg_insertion.train_rlt --eval_only --checkpoint path/to/ckpt.pkl

  # Test mode (no hardware):
  python -m rlt.examples.peg_insertion.train_rlt --fake_env
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import pickle as pkl
from pathlib import Path

import numpy as np

# Force JAX to CPU so SAC doesn't compete with VLA server for GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add paths
sys.path.insert(0, str(Path(__file__).parents[3]))

from rlt.examples.peg_insertion.config import RLTConfig, RLT_CONFIG_MAPPING
from rlt.agents.rlt_buffer import RLTBuffer
from rlt.envs.ur5e_rlt_env import UR5eRLTEnv


def load_vla_client(config: RLTConfig):
    """Create VLA WebSocket client that connects to the running VLA server.
    
    The VLA server runs separately (in openpi venv, Python 3.11):
        cd openpi_ur5e/openpi-ur5e
        .venv/bin/python scripts/serve_policy.py --port 8000 ...
    
    This client connects via websocket — no need for JAX/OpenPI deps here.
    """
    from rlt.models.vla_client import VLAClient
    
    try:
        client = VLAClient(
            server_url=f"ws://localhost:{config.vla_server_port}",
            prompt=config.language_instruction,
            action_horizon=config.vla_action_horizon,
            action_dim=7,  # 6 joints + 1 gripper (server returns 7D)
        )
        if client.is_connected():
            print(f"[RLT] VLA client connected to ws://localhost:{config.vla_server_port}")
            return client
        else:
            print("[RLT] WARNING: VLA client could not connect")
            return None
    except Exception as e:
        print(f"[RLT] WARNING: VLA client creation failed: {e}")
        return None


def load_rl_token_model(config: RLTConfig, device: str = "cpu"):
    """Load trained RL Token encoder.
    
    Always loads on CPU first to avoid CUDA context issues,
    then optionally moves to GPU.
    """
    import torch
    from rlt.models.rl_token import RLTokenModel

    ckpt_path = Path(config.rl_token_checkpoint)
    if not ckpt_path.exists():
        print(f"[RLT] WARNING: RL Token checkpoint not found: {ckpt_path}")
        print("[RLT] Running without RL Token (zero embeddings)")
        return None

    # Always load on CPU first to avoid CUDA issues
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    model = RLTokenModel(
        embed_dim=cfg["embed_dim"],
        token_dim=cfg["token_dim"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        n_heads=cfg.get("n_heads", 8),
        ffn_dim=cfg.get("ffn_dim", 2048),
        max_len=cfg.get("max_len", 958),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Move to GPU only if requested AND available
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"[RLT] RL Token model on CUDA")
    else:
        print(f"[RLT] RL Token model on CPU")
    
    print(f"[RLT] Loaded RL Token model (loss={ckpt['loss']:.5f}, step={ckpt['step']})")
    return model


def create_sac_agent(config: RLTConfig):
    """Create the SAC agent for residual learning.
    
    SAC runs on JAX CPU (lightweight MLPs, no GPU needed).
    """
    from rlt.agents.sac_agent import RLTSACAgent

    obs_dim = config.token_dim + config.proprio_dim + (config.chunk_size * config.action_dim)
    action_dim = config.chunk_size * config.action_dim  # residual for full chunk

    agent = RLTSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(config.hidden_dims),
        ensemble_size=config.critic_ensemble_size,
        subsample_size=config.critic_subsample_size,
        discount=config.discount,
        tau=config.soft_target_update_rate,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        temp_lr=config.temp_lr,
        beta=config.beta,
    )
    print(f"[RLT] SAC agent created (obs={obs_dim}, action={action_dim}, device=cpu/jax)")
    return agent


def run_warmup_episodes(
    env: UR5eRLTEnv,
    config: RLTConfig,
    online_buffer: RLTBuffer,
) -> list:
    """Run VLA-only episodes to fill the online buffer.

    During warmup, the residual is zero — we just execute VLA actions.
    This gives the RL agent some initial experience to learn from.
    """
    print(f"\n{'='*60}")
    print(f"  WARMUP: {config.warmup_episodes} VLA-only episodes")
    print(f"{'='*60}")

    results = []
    for ep in range(config.warmup_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        z_rl_list = []
        proprio_list = []
        action_list = []
        ref_list = []

        while not done:
            # VLA-only: residual = 0
            residual = np.zeros(config.chunk_size * config.action_dim, dtype=np.float32)

            # Record current state
            z_rl_list.append(obs[:config.token_dim].copy())
            proprio_list.append(
                obs[config.token_dim:config.token_dim + config.proprio_dim].copy()
            )

            # Extract reference from obs
            ref_start = config.token_dim + config.proprio_dim
            ref_flat = obs[ref_start:]
            ref_chunk = ref_flat.reshape(config.chunk_size, config.action_dim)

            obs, reward, terminated, truncated, info = env.step(residual)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            # Store per-step actions (reference only during warmup)
            for i in range(config.chunk_size):
                action_list.append(ref_chunk[i].copy())
                ref_list.append(ref_chunk[i].copy())

        # Add episode to online buffer
        if z_rl_list:
            T = min(len(action_list), len(z_rl_list) * config.chunk_size)
            z_rl_expanded = []
            proprio_expanded = []
            for z, p in zip(z_rl_list, proprio_list):
                for _ in range(config.chunk_size):
                    z_rl_expanded.append(z)
                    proprio_expanded.append(p)

            added = online_buffer.add_episode(
                z_rl_list=z_rl_expanded[:T],
                proprio_list=proprio_expanded[:T],
                action_list=action_list[:T],
                ref_list=ref_list[:T],
                reward=ep_reward,
                done=True,
            )

            results.append(ep_reward > 0)
            success_str = "\u2713" if ep_reward > 0 else "\u2717"
            print(f"  Warmup ep {ep+1:3d}/{config.warmup_episodes}: "
                  f"{success_str} reward={ep_reward:.1f} "
                  f"chunks={ep_steps} +{added} trans")

    sr = sum(results) / len(results) if results else 0
    print(f"\n  Warmup done: {sum(results)}/{len(results)} success ({sr:.0%})")
    print(f"  Online buffer: {len(online_buffer)} transitions")
    return results


def run_training(
    env: UR5eRLTEnv,
    config: RLTConfig,
    online_buffer: RLTBuffer,
    demo_buffer: RLTBuffer,
    agent=None,
):
    """Main RLT training loop with SAC.

    The agent learns residual corrections to the VLA reference actions.
    """
    print(f"\n{'='*60}")
    print(f"  RLT TRAINING: {config.total_episodes} episodes")
    print(f"  Task: {config.task_name}")
    print(f"  BC weight \u03b2={config.beta}, ref_dropout={config.ref_dropout}")
    print(f"  UTD={config.utd_ratio}, batch={config.batch_size}")
    print(f"  Agent: {'SAC' if agent else 'RANDOM (no agent)'}")
    print(f"{'='*60}\n")

    results = []
    total_grad_steps = 0
    best_sr = 0.0

    for episode in range(config.total_episodes):
        # ── Rollout ──────────────────────────────────────────────────────
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_residual_norms = []

        z_rl_list = []
        proprio_list = []
        action_list = []
        ref_list = []

        while not done:
            # Record current state
            z_rl = obs[:config.token_dim].copy()
            proprio = obs[config.token_dim:config.token_dim + config.proprio_dim].copy()
            ref_start = config.token_dim + config.proprio_dim
            ref_chunk = obs[ref_start:].reshape(config.chunk_size, config.action_dim)

            z_rl_list.append(z_rl)
            proprio_list.append(proprio)

            # ── RL Policy: sample residual action ────────────────────────
            if agent is not None:
                residual = agent.sample_action(obs, deterministic=False)
            else:
                # No agent — small random exploration
                residual = np.random.randn(
                    config.chunk_size * config.action_dim
                ).astype(np.float32) * 0.1

            # Step
            obs, reward, terminated, truncated, info = env.step(residual)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            ep_residual_norms.append(info.get("residual_norm", 0))

            # Store per-step data
            residual_chunk = residual.reshape(config.chunk_size, config.action_dim)
            for i in range(config.chunk_size):
                final_action = ref_chunk[i] + residual_chunk[i] * np.array(
                    [config.max_residual_pos]*3 + [config.max_residual_rot]*3
                )
                action_list.append(final_action)
                ref_list.append(ref_chunk[i].copy())

        # ── Store episode ────────────────────────────────────────────────
        T = min(len(action_list), len(z_rl_list) * config.chunk_size)
        z_rl_expanded = []
        proprio_expanded = []
        for z, p in zip(z_rl_list, proprio_list):
            for _ in range(config.chunk_size):
                z_rl_expanded.append(z)
                proprio_expanded.append(p)

        if T > 0:
            online_buffer.add_episode(
                z_rl_list=z_rl_expanded[:T],
                proprio_list=proprio_expanded[:T],
                action_list=action_list[:T],
                ref_list=ref_list[:T],
                reward=ep_reward,
                done=True,
            )

        # ── Gradient updates ─────────────────────────────────────────────
        if agent is not None and online_buffer.is_ready(config.training_starts):
            for _ in range(config.utd_ratio):
                batch = online_buffer.sample_rlpd(
                    config.batch_size,
                    demo_buffer=demo_buffer if len(demo_buffer) > 0 else None,
                    demo_ratio=config.demo_ratio,
                )
                update_info = agent.update(batch)
                total_grad_steps += 1

        # ── Logging ──────────────────────────────────────────────────────
        results.append(ep_reward > 0)
        success_str = "\u2713" if ep_reward > 0 else "\u2717"

        if (episode + 1) % config.log_interval == 0:
            recent = results[-config.log_interval:]
            sr = sum(recent) / len(recent)
            avg_res = np.mean(ep_residual_norms) if ep_residual_norms else 0
            log_msg = (
                f"  Ep {episode+1:4d}/{config.total_episodes} | "
                f"SR(last {config.log_interval})={sr:.0%} | "
                f"{success_str} r={ep_reward:.1f} | "
                f"chunks={ep_steps} | "
                f"residual={avg_res:.4f} | "
                f"buf={len(online_buffer):,} | "
                f"grad={total_grad_steps:,}"
            )
            if agent is not None and total_grad_steps > 0:
                log_msg += f" | alpha={float(np.exp(agent.log_alpha)):.3f}"
            print(log_msg)

        # ── Save checkpoint ──────────────────────────────────────────────
        if (episode + 1) % config.save_interval == 0:
            ckpt_dir = Path(config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"rlt_ep{episode+1:04d}.pkl"
            if agent is not None:
                agent.save(str(ckpt_path))
            else:
                print(f"  [Checkpoint] \u2192 {ckpt_path} (no agent to save)")

            # Save best
            recent_50 = results[-50:]
            current_sr = sum(recent_50) / len(recent_50)
            if current_sr > best_sr and agent is not None:
                best_sr = current_sr
                best_path = ckpt_dir / "best.pkl"
                agent.save(str(best_path))
                print(f"  [Best] SR={best_sr:.0%} \u2192 {best_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total episodes: {len(results)}")
    final_sr = sum(results[-50:]) / min(50, len(results))
    print(f"  Final success rate (last 50): {final_sr:.0%}")
    print(f"  Best success rate: {best_sr:.0%}")
    print(f"  Total gradient steps: {total_grad_steps:,}")
    print(f"  Online buffer: {len(online_buffer):,} transitions")
    print(f"{'='*60}")


def run_eval(
    env: UR5eRLTEnv,
    config: RLTConfig,
    agent,
    num_episodes: int = 20,
    no_residual: bool = False,
):
    """Evaluate the trained agent (or VLA-only baseline)."""
    mode = "VLA-only" if no_residual else "VLA + RLT (SAC)"
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {mode}")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            if no_residual or agent is None:
                residual = np.zeros(
                    config.chunk_size * config.action_dim, dtype=np.float32
                )
            else:
                residual = agent.sample_action(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(residual)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        success = ep_reward > 0
        results.append(success)
        s = "\u2713" if success else "\u2717"
        print(f"  Eval ep {ep+1:3d}/{num_episodes}: {s} reward={ep_reward:.1f} chunks={ep_steps}")

    sr = sum(results) / len(results)
    print(f"\n  {'='*40}")
    print(f"  RESULT: {sum(results)}/{len(results)} = {sr:.0%} success rate")
    print(f"  Mode: {mode}")
    print(f"  {'='*40}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RLT Training \u2014 Peg Insertion")
    parser.add_argument("--task", default="peg_insertion",
                        choices=list(RLT_CONFIG_MAPPING.keys()))
    parser.add_argument("--fake_env", action="store_true",
                        help="Run without hardware (for testing)")
    parser.add_argument("--no_vla", action="store_true",
                        help="Skip VLA connection (random reference)")
    parser.add_argument("--no_rl", action="store_true",
                        help="Disable RL agent (random residuals)")
    parser.add_argument("--no_rl_token", action="store_true",
                        help="Skip RL Token model (zero embeddings)")
    parser.add_argument("--device", default="cpu",
                        help="Device for RL Token model (cpu recommended)")
    parser.add_argument("--warmup_only", action="store_true",
                        help="Only run warmup episodes (VLA-only baseline)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Evaluate a checkpoint (no training)")
    parser.add_argument("--no_residual", action="store_true",
                        help="Evaluate with zero residual (VLA-only)")
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to agent checkpoint for eval")
    parser.add_argument("--rl_token_ckpt", type=str, default=None,
                        help="Override RL Token checkpoint path")
    args = parser.parse_args()

    # Load config
    config = RLT_CONFIG_MAPPING[args.task]()

    # Override RL Token checkpoint if specified
    if args.rl_token_ckpt:
        config.rl_token_checkpoint = args.rl_token_ckpt

    sep = '\u2550' * 60
    print(f"\n{sep}")
    print(f"  RLT-UR5e \u2014 {config.task_name.upper()}")
    print(f"{sep}")
    print(f"  VLA server: ws://localhost:{config.vla_server_port}")
    print(f"  RL Token: {config.rl_token_checkpoint}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Max residual: pos={config.max_residual_pos*1000:.1f}mm, "
          f"rot={np.degrees(config.max_residual_rot):.1f}\u00b0")
    if args.eval_only:
        print(f"  Mode: EVALUATION ({'VLA-only' if args.no_residual else 'VLA+RLT'})")
    elif args.warmup_only:
        print(f"  Mode: VLA-ONLY BASELINE (warmup)")
    else:
        print(f"  Mode: FULL RLT TRAINING")
    print()

    # ── Load VLA client (websocket to server) ────────────────────────────
    vla_client = None
    if not args.no_vla and not args.fake_env:
        vla_client = load_vla_client(config)

    # ── Load RL Token model ──────────────────────────────────────────────
    rl_token_model = None
    if not args.no_rl_token and not args.fake_env:
        rl_token_model = load_rl_token_model(config, device=args.device)

    # ── Create environment ───────────────────────────────────────────────
    env = UR5eRLTEnv(
        config=config,
        vla_client=vla_client,
        rl_token_model=rl_token_model,
        fake_env=args.fake_env,
    )
    print(f"[RLT] Environment created (fake={args.fake_env})")
    print(f"[RLT] Obs space: {env.observation_space.shape}")
    print(f"[RLT] Action space: {env.action_space.shape}")

    # ── Create SAC agent ─────────────────────────────────────────────────
    agent = None
    if not args.no_rl and not args.warmup_only:
        agent = create_sac_agent(config)

        # Load checkpoint if specified
        if args.checkpoint and Path(args.checkpoint).exists():
            agent.load(args.checkpoint)

    # ── Evaluation mode ──────────────────────────────────────────────────
    if args.eval_only:
        run_eval(
            env, config, agent,
            num_episodes=args.eval_episodes,
            no_residual=args.no_residual,
        )
        env.close()
        return

    # ── Create buffers ───────────────────────────────────────────────────
    online_buffer = RLTBuffer(
        capacity=config.replay_buffer_capacity,
        token_dim=config.token_dim,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
    )

    demo_buffer = RLTBuffer(
        capacity=50_000,
        token_dim=config.token_dim,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
    )

    print(f"[RLT] Buffers created (online={config.replay_buffer_capacity:,}, demo=50k)")

    # ── Warmup ───────────────────────────────────────────────────────────
    warmup_results = run_warmup_episodes(env, config, online_buffer)

    if args.warmup_only:
        print(f"\n[RLT] VLA-only baseline: {sum(warmup_results)}/{len(warmup_results)} success")
        env.close()
        return

    # ── Main training loop ───────────────────────────────────────────────
    run_training(env, config, online_buffer, demo_buffer, agent=agent)

    # ── Cleanup ──────────────────────────────────────────────────────────
    env.close()
    print("\n[RLT] Done. Robot safe. \U0001f916")


if __name__ == "__main__":
    main()
