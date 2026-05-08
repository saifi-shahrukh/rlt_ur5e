# 02 — UV Environment Setup

## Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

## Create Virtual Environment

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl

# Create venv with Python 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
```

## Install Packages

```bash
# Core packages (editable installs)
uv pip install -e .
uv pip install -e serl_launcher/

# JAX with CUDA 12 (CRITICAL — must match your CUDA version)
uv pip install --upgrade "jax[cuda12]==0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Additional dependencies
uv pip install tensorflow tf-keras ml_collections matplotlib natsort wandb
uv pip install pyrealsense2 freenect2 pynput
uv pip install ur-rtde
```

## Verify Installation

```bash
# Check JAX sees GPU
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0)]

# Check imports work
python -c "from serl_launcher.agents.continuous.sac import SACAgent; print('✅ serl_launcher OK')"
python -c "from ur_env.envs.ur5e_env import UR5eEnv; print('✅ ur_env OK')"

# Run tests
python -m pytest tests/ -q --ignore=tests/test_pipeline.py
```

## PYTHONPATH Setup

When running examples, always set:

```bash
cd ~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl/examples
source ../.venv/bin/activate
export PYTHONPATH="../serl_robot_infra:.:$PYTHONPATH"
```

Or add to your `.bashrc`:
```bash
export SERL_ROOT=~/ur5e_hande_workspace/serl_setup/ur5e_hil_serl
alias serl-activate='cd $SERL_ROOT/examples && source $SERL_ROOT/.venv/bin/activate && export PYTHONPATH="$SERL_ROOT/serl_robot_infra:.:$PYTHONPATH"'
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `jax.devices()` shows `[CpuDevice]` | JAX CUDA not installed correctly. Reinstall with matching CUDA version |
| `ModuleNotFoundError: ur_env` | Set PYTHONPATH to include `serl_robot_infra` |
| `freenect2` import error | Rebuild libfreenect2 and reinstall Python bindings |
| `pyrealsense2` import error | `pip install pyrealsense2` or build from source |
