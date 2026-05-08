# OpenPI UR5e Setup Guide — RTX 5070 Ti + RealSense D435 + Kinect

## Hardware
- **Robot**: UR5e + Robotiq Hand-E gripper
- **GPU**: NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120)
- **Wrist Camera**: Intel RealSense D435i (serial: 034422070605)
- **Third-Person Camera**: Xbox Kinect v2 (serial: 000631452147)
- **Robot IP**: 172.22.1.139
- **OS**: Ubuntu, Python 3.11.15

## Directory Structure

~/workspace/ur5e_workspace/openpi_ur5e/ ├── openpi-ur5e/ # OpenPI fork (F-Fer) — policy server │ ├── src/openpi/ # Core library │ ├── scripts/ # serve_policy.py, train.py, etc. │ ├── assets/ # Normalization stats (NOT checkpoints) │ ├── checkpoints/ # Downloaded/trained model weights │ │ └── F-Fer/tasks-merged-lora/59999/ # LoRA fine-tuned checkpoint │ └── .venv/ # Python virtual environment └── lerobot_ur5e_gello/ # Robot teleoperation + data collection

## Setup Steps (Completed)

### 1. Clone & Install
```bash
cd ~/workspace/ur5e_workspace/openpi_ur5e
git clone https://github.com/F-Fer/openpi-ur5e.git
cd openpi-ur5e
git submodule update --init --recursive
uv venv
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

2. Fix PyTorch for RTX 5070 Ti (sm_120)

Default PyTorch only supports up to sm_90. Must use nightly:

uv pip uninstall torch torchvision triton
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

Result: torch==2.12.0.dev20260407+cu128
3. Verify GPU

### PyTorch

python -c "import torch; x=torch.randn(3,3).cuda(); print('PyTorch GPU OK')"

### JAX (used by inference server)

python -c "import jax; print(jax.devices()); x=jax.numpy.ones((100,100)); print((x@x).shape)"

Both must show GPU device. JAX shows [CudaDevice(id=0)].
4. Fix SSL (Docker cert not available on bare metal)
Edit src/openpi/serving/websocket_policy_server.py:

Line 40: ssl_context = None  # SSL disabled for local dev
Line 42: # ssl_context.load_cert_chain (disabled for local dev)

Line 51 (ssl=ssl_context) now passes None → plain ws:// instead of wss://.
5. Download Fine-Tuned Checkpoint

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='F-Fer/tasks-merged-lora',
    local_dir='./checkpoints/F-Fer/tasks-merged-lora',
)
"

Running the Models
Base Model (Zero-Shot, no training needed)

source .venv/bin/activate
uv run scripts/serve_policy.py --env UR5E --port 8000

    Downloads gs://openpi-assets/checkpoints/pi0_base (~6 GB, cached at ~/.cache/openpi/)
    Uses config pi0_ur_zero_shot

Fine-Tuned Model (LoRA, all UR5e tasks)

source .venv/bin/activate
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_ur_tasks_merged_lora \
  --policy.dir ./checkpoints/F-Fer/tasks-merged-lora/59999

    Loads F-Fer's LoRA checkpoint (step 59999)
    Norm stats from assets/pi0_ur_tasks_merged_lora/F-Fer/ur-tasks-merged

Test Client (in separate terminal)

source .venv/bin/activate
python << 'EOF'
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="localhost", port=8000)
obs = {
    "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros(6, dtype=np.float32),
    "observation/gripper_position": np.zeros(1, dtype=np.float32),
    "prompt": "pick up the tissue box",
}
result = client.infer(obs)
print(f"Actions: {np.array(result['actions']).shape}")  # (30, 7)
EOF

Key API Reference
Client Import

from openpi_client.websocket_client_policy import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=8000)

Client Methods

result = client.infer(obs)          # Returns {"actions": ndarray, "policy_timing": ..., "server_timing": ...}
client.reset()                      # Reset policy state
client.get_server_metadata()        # Get server info

Action Output

    Shape: (30, 7) → 30-step horizon, 7 dims (6 joints + 1 gripper)
    Actions are delta (relative) joint positions

Camera → Observation Key Mapping
Hardware	Observation Key	Size
RealSense D435 (wrist)	observation/wrist_image_left	224×224×3 uint8
Kinect (third-person)	observation/exterior_image_1_left	224×224×3 uint8
Not present	observation/wrist_image_right	zeros (masked)
UR5e joints	observation/joint_position	6 float32
Hand-E gripper	observation/gripper_position	1 float32
Available Configs & Checkpoints
On Google Cloud (base models)
GCS Checkpoint	Description
pi0_base	π₀ base model (used for zero-shot and as fine-tune base)
pi05_base	π₀.5 base model
pi0_fast_base	π₀-FAST base model
On HuggingFace (F-Fer fine-tuned)
Config Name	HF Model	Description
pi0_ur_tasks_merged_lora	F-Fer/tasks-merged-lora	LoRA, all tasks merged ✅
pi0_ur_tasks_merged	F-Fer/tasks-merged	Full fine-tune, all tasks
pi0_ur_task1_lora	F-Fer/task1-lora	LoRA, task 1
pi0_ur_task2_lora	F-Fer/task2-lora	LoRA, task 2
pi0_ur_task3_lora	F-Fer/task3-lora	LoRA, task 3
pi0_ur_task4_lora	F-Fer/task4-lora	LoRA, task 4
Teleoperation (GELLO)

cd ~/workspace/ur5e_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate
./scripts/record_wrapper.sh \
    --robot.type=ur5e \
    --robot.ip=172.22.1.139 \
    --dataset.repo_id=saifi/ur5e-peg-insertion \
    --dataset.single_task="Pick up the bolt and insert it into the hole."

Known Issues

    rocm and tpu backend warnings are harmless (INFO level, not errors)
    Norm stats not found in assets/pi0_ur_zero_shot/... for zero-shot is expected (falls back to cached base model stats)
    PyTorch version incompatibility warnings with lerobot and openpi pinned versions — does not affect JAX inference server

Save this file:

```bash
cd ~/workspace/ur5e_workspace/openpi_ur5e/openpi-ur5e
cat > SETUP_README.md << 'READMEEOF'
# (paste the above content)
READMEEOF

Phase 3: Connect Fine-Tuned Model to Real Robot
Now we need to bridge the inference server (port 8000) with your real robot. First, let's understand your GELLO setup:

cd ~/workspace/ur5e_workspace/openpi_ur5e/lerobot_ur5e_gello
source .venv/bin/activate

# Find the policy execution scripts
find . -name "*.py" -exec grep -l "openpi\|WebsocketClient\|infer\|serve" {} \;

# Find the record/replay scripts
ls scripts/
cat scripts/record_wrapper.sh | head -50

# Find any evaluation/deployment scripts
find . -name "*.py" -exec grep -l "eval\|deploy\|rollout\|episode" {} \;

Also check:

# See if there's already an openpi integration
grep -rn "openpi\|8000\|websocket\|policy.*server" . --include="*.py" | head -20

# Check the robot control loop
find . -name "*.py" -exec grep -l "ur5e\|UR5\|rtde\|urx" {} \; | head -10

Paste the output and I'll write the exact script to connect your fine-tuned model to the real UR5e through GELLO.
Sources:

    https://github.com/F-Fer/openpi-ur5e.git 🌐
    https://download.pytorch.org/whl/nightly/cu128 🌐