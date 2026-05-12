#!/usr/bin/env python3
"""
Benchmark Script — Systematic evaluation of VLA-only vs VLA+RLT vs SERL.

This script runs controlled evaluations for paper-quality benchmarking:
  1. VLA-only baseline (zero residual, just execute VLA actions)
  2. VLA + RLT (SAC residual corrections)
  3. SERL baseline (standard HIL-SERL from scratch, no VLA)

Usage:
  cd ~/ur5e_hande_workspace/rlt_ur5e
  source ur5e_hil_serl/.venv/bin/activate
  export JAX_PLATFORMS=cpu
  export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH"

  # Evaluate VLA-only (requires VLA server running on port 8000):
  python scripts/benchmark.py --mode vla_only --episodes 20

  # Evaluate VLA+RLT (requires VLA server + trained RLT checkpoint):
  python scripts/benchmark.py --mode rlt --episodes 20 --checkpoint checkpoints/rlt_runs/peg_insertion/best.pkl

  # Evaluate SERL baseline (uses HIL-SERL trained checkpoint):
  python scripts/benchmark.py --mode serl --episodes 20 --serl_checkpoint path/to/serl.pkl

  # Run all modes and generate comparison report:
  python scripts/benchmark.py --mode all --episodes 20 --checkpoint checkpoints/rlt_runs/peg_insertion/best.pkl

  # Fake env test:
  python scripts/benchmark.py --mode all --episodes 10 --fake_env

Outputs:
  - Console summary table
  - JSON results file in results/benchmark_<timestamp>.json
  - CSV for plotting
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlt.examples.peg_insertion.config import RLTConfig
from rlt.envs.ur5e_rlt_env import UR5eRLTEnv
from rlt.agents.sac_agent import RLTSACAgent


def evaluate_vla_only(env, config, num_episodes: int) -> dict:
    """Evaluate VLA-only policy (zero residual)."""
    results = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        
        while not done:
            residual = np.zeros(config.chunk_size * config.action_dim, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(residual)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
        
        results.append(ep_reward > 0)
        episode_lengths.append(ep_steps)
        s = "\u2713" if ep_reward > 0 else "\u2717"
        print(f"  [{ep+1:3d}/{num_episodes}] {s} reward={ep_reward:.1f} chunks={ep_steps}")
    
    return {
        "mode": "VLA-only",
        "success_rate": sum(results) / len(results),
        "successes": sum(results),
        "total": len(results),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "results": results,
    }


def evaluate_rlt(env, config, agent, num_episodes: int) -> dict:
    """Evaluate VLA + RLT (SAC residual)."""
    results = []
    episode_lengths = []
    residual_norms = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_res_norms = []
        
        while not done:
            residual = agent.sample_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(residual)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            ep_res_norms.append(info.get("residual_norm", 0))
        
        results.append(ep_reward > 0)
        episode_lengths.append(ep_steps)
        residual_norms.append(np.mean(ep_res_norms))
        s = "\u2713" if ep_reward > 0 else "\u2717"
        print(f"  [{ep+1:3d}/{num_episodes}] {s} reward={ep_reward:.1f} chunks={ep_steps} res_norm={np.mean(ep_res_norms):.4f}")
    
    return {
        "mode": "VLA+RLT",
        "success_rate": sum(results) / len(results),
        "successes": sum(results),
        "total": len(results),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_residual_norm": float(np.mean(residual_norms)),
        "results": results,
    }


def evaluate_serl(env, config, num_episodes: int, serl_checkpoint: str = None) -> dict:
    """Evaluate standard SERL baseline (no VLA).
    
    This uses the HIL-SERL agent directly on the SERL environment
    (bypassing the VLA entirely).
    """
    # TODO: Load SERL agent from checkpoint and evaluate
    # For now, return placeholder
    print("  [SERL] Evaluation not yet implemented (requires SERL checkpoint)")
    return {
        "mode": "SERL",
        "success_rate": 0.0,
        "successes": 0,
        "total": num_episodes,
        "note": "Not implemented yet",
    }


def print_comparison_table(all_results: list[dict]):
    """Print a comparison table."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPARISON — Peg Insertion")
    print(f"{'='*70}")
    print(f"  {'Mode':<20} {'Success Rate':>15} {'Episodes':>10} {'Avg Length':>12}")
    print(f"  {'-'*20} {'-'*15} {'-'*10} {'-'*12}")
    
    for r in all_results:
        sr_str = f"{r['success_rate']:.0%} ({r['successes']}/{r['total']})"
        length_str = f"{r.get('mean_episode_length', 0):.1f}" if 'mean_episode_length' in r else "N/A"
        print(f"  {r['mode']:<20} {sr_str:>15} {r['total']:>10} {length_str:>12}")
    
    print(f"{'='*70}")
    
    # Statistical comparison
    if len(all_results) >= 2:
        vla = next((r for r in all_results if r['mode'] == 'VLA-only'), None)
        rlt = next((r for r in all_results if r['mode'] == 'VLA+RLT'), None)
        if vla and rlt:
            improvement = rlt['success_rate'] - vla['success_rate']
            print(f"\n  RLT improvement over VLA-only: {improvement:+.0%}")
            if vla['success_rate'] > 0:
                relative = improvement / vla['success_rate']
                print(f"  Relative improvement: {relative:+.0%}")
    print()


def save_results(all_results: list[dict], output_dir: str = "results"):
    """Save results to JSON and CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = Path(output_dir) / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Results saved: {json_path}")
    
    # CSV
    csv_path = Path(output_dir) / f"benchmark_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("mode,success_rate,successes,total,mean_episode_length\n")
        for r in all_results:
            f.write(f"{r['mode']},{r['success_rate']:.4f},{r['successes']},{r['total']},{r.get('mean_episode_length', 0):.1f}\n")
    print(f"  CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLA vs RLT vs SERL")
    parser.add_argument("--mode", choices=["vla_only", "rlt", "serl", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task", default="peg_insertion")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rlt_runs/peg_insertion/best.pkl")
    parser.add_argument("--serl_checkpoint", type=str, default=None)
    parser.add_argument("--fake_env", action="store_true")
    parser.add_argument("--no_vla", action="store_true", help="Skip VLA server connection")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    
    config = RLTConfig()
    
    # Connect to VLA server
    vla_client = None
    if not args.no_vla and not args.fake_env:
        from rlt.models.vla_client import VLAClient
        vla_client = VLAClient(
            server_url=f"ws://localhost:{config.vla_server_port}",
            prompt=config.language_instruction,
        )
        if not vla_client.is_connected():
            print("WARNING: VLA server not available. Running without VLA.")
            vla_client = None
    
    # Create environment
    env = UR5eRLTEnv(
        config=config,
        vla_client=vla_client,
        rl_token_model=None,
        fake_env=args.fake_env,
    )
    
    all_results = []
    
    # ── VLA-only ─────────────────────────────────────────────────────────
    if args.mode in ("vla_only", "all"):
        print(f"\n{'─'*50}")
        print(f"  Evaluating: VLA-only ({args.episodes} episodes)")
        print(f"{'─'*50}")
        result = evaluate_vla_only(env, config, args.episodes)
        all_results.append(result)
    
    # ── VLA + RLT ────────────────────────────────────────────────────────
    if args.mode in ("rlt", "all"):
        print(f"\n{'─'*50}")
        print(f"  Evaluating: VLA+RLT ({args.episodes} episodes)")
        print(f"{'─'*50}")
        
        # Load SAC agent
        agent = RLTSACAgent(
            obs_dim=config.token_dim + config.proprio_dim + config.chunk_size * config.action_dim,
            action_dim=config.chunk_size * config.action_dim,
            hidden_dims=tuple(config.hidden_dims),
        )
        
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            agent.load(str(ckpt_path))
            result = evaluate_rlt(env, config, agent, args.episodes)
        else:
            print(f"  WARNING: Checkpoint not found: {ckpt_path}")
            print(f"  Running with untrained agent (random residuals)")
            result = evaluate_rlt(env, config, agent, args.episodes)
        all_results.append(result)
    
    # ── SERL ─────────────────────────────────────────────────────────────
    if args.mode in ("serl", "all"):
        print(f"\n{'─'*50}")
        print(f"  Evaluating: SERL baseline ({args.episodes} episodes)")
        print(f"{'─'*50}")
        result = evaluate_serl(env, config, args.episodes, args.serl_checkpoint)
        all_results.append(result)
    
    # ── Comparison ───────────────────────────────────────────────────────
    if all_results:
        print_comparison_table(all_results)
        save_results(all_results, args.output_dir)
    
    env.close()


if __name__ == "__main__":
    main()
