#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Cache the FAST tokenizer on headnode (has internet)
# Compute nodes have NO internet → must be pre-cached
# ═══════════════════════════════════════════════════════════════════════════════
set -e

VENV=/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv
PYTHON="${VENV}/bin/python3.11"

echo "═══════════════════════════════════════════════════════════════"
echo "  Caching physical-intelligence/fast tokenizer"
echo "  (Required for π0-FAST training)"
echo "═══════════════════════════════════════════════════════════════"

# Download and cache the FAST tokenizer
${PYTHON} -c "
from transformers import AutoProcessor
import os

print('Downloading physical-intelligence/fast...')
proc = AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True)
print(f'✓ Cached at: {os.environ.get(\"HF_HOME\", \"~/.cache/huggingface\")}')
print('  Compute nodes can now use HF_HUB_OFFLINE=1')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ FAST tokenizer cached! π0-FAST can now run offline."
echo "═══════════════════════════════════════════════════════════════"
