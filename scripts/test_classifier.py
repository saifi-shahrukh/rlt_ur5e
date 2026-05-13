#!/usr/bin/env python3
"""Test the reward classifier by running the SERL env with keyboard control.

This bypasses the RLT pipeline entirely — just raw SERL env + keyboard.
Use this to verify:
1. Keyboard moves the robot
2. Classifier fires when peg is inserted

Run from: /home/robolab-2/ur5e_hande_workspace/rlt_ur5e
With: source ur5e_hil_serl/.venv/bin/activate && export JAX_PLATFORMS=cpu && \
      export PYTHONPATH="$PWD:$PWD/ur5e_hil_serl:$PWD/ur5e_hil_serl/serl_robot_infra:$PWD/ur5e_hil_serl/examples:$PYTHONPATH" && \
      python scripts/test_classifier.py
"""
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ur5e_hil_serl'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ur5e_hil_serl', 'serl_robot_infra'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ur5e_hil_serl', 'examples'))

from experiments.mappings import CONFIG_MAPPING

print("="*60)
print("  CLASSIFIER TEST — Manual Keyboard Control")
print("="*60)
print()
print("  This tests the SERL env directly with keyboard.")
print("  Use arrow keys to move, try to insert peg.")
print("  Watch for [CLASSIFIER] REWARD=1 messages.")
print("  Press Ctrl+C to stop.")
print()
print("="*60)

# Create env with classifier
config = CONFIG_MAPPING['peg_insertion']()
env = config.get_environment(fake_env=False, save_video=False, classifier=True)

print("\n[TEST] Environment created with classifier")
print(f"[TEST] Action space: {env.action_space.shape}")
print(f"[TEST] Observation space keys: {list(env.observation_space.spaces.keys()) if hasattr(env.observation_space, 'spaces') else env.observation_space.shape}")

# Run episodes
for ep in range(10):
    print(f"\n{'='*40}")
    print(f"  Episode {ep+1}/10 — Use keyboard to insert peg!")
    print(f"  Arrow keys = XY, 1/0 = Z up/down")
    print(f"{'='*40}")
    
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    while not done:
        # Send zero action — keyboard intervention will override
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        total_reward += reward
        
        # Print ABSOLUTE TCP position directly from env internals
        if step_count % 20 == 0:
            try:
                # Access the raw absolute TCP from base env
                base = env
                while hasattr(base, 'env'):
                    base = base.env
                abs_tcp = base.currpos[:3]  # absolute xyz in meters
                target = np.array([0.36066, 0.08130, 0.090])
                dist = np.linalg.norm(abs_tcp - target) * 1000  # mm
                print(f"  step={step_count:3d} | TCP=[{abs_tcp[0]:.4f},{abs_tcp[1]:.4f},{abs_tcp[2]:.4f}]m | dist_to_target={dist:.1f}mm")
            except Exception as e:
                if 'state' in obs:
                    state = obs['state']
                    if hasattr(state, 'shape') and state.ndim > 1:
                        state = state[0]
                    print(f"  step={step_count:3d} | state[:6]={state[:6]} | reward={reward}")
                else:
                    print(f"  step={step_count:3d} | reward={reward}")
        
        if reward > 0:
            print(f"\n  *** SUCCESS! reward={reward} at step {step_count} ***")
            break
    
    status = "✓ SUCCESS" if total_reward > 0 else "✗ FAILED"
    print(f"\n  Episode {ep+1} done: {status} | steps={step_count} | reward={total_reward}")
    
    if total_reward > 0:
        print("\n  Classifier is working! You can use it for RLT training.")
        break

print("\n[TEST] Done. Ctrl+C to exit.")
