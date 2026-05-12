# Checkpoint Transfer Guide

## Network Topology

    HPC Cluster (hpc-headnode.iis.fhg.de)
        |  Internal Fraunhofer network
        |  SSH: saifi@hpc-headnode.iis.fhg.de
        v
    WSL2 Ubuntu on Windows (hostname: r10028)
        |  IP: 172.20.234.194 (WSL2 eth0)
        |  Can reach both HPC and local network
        v
    Physical Ubuntu (hostname: robolab-2)
        |  IP: 172.22.1.188
        |  Robot workstation with GPU
        v
    UR5e Robot + Cameras

Direct HPC -> Physical Ubuntu is NOT possible (different network segments).
WSL2 acts as the bridge.

---

## Transfer: VLA Checkpoints

### Step 1: HPC -> WSL2

Run from WSL2 terminal:

    # Create local directory
    mkdir -p ~/hpc_checkpoints

    # Full checkpoints (all 3 models):
    rsync -avz --progress \
      saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/ \
      ~/hpc_checkpoints/

    # Or params-only (much smaller, sufficient for inference):
    for config in pi0_ur5e_peg_insertion_lora pi05_ur5e_peg_insertion_lora pi0_fast_ur5e_peg_insertion_lora; do
      STEP=$(ssh saifi@hpc-headnode.iis.fhg.de "ls /data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/${config}/peg_insertion_50demos/ | sort -V | tail -1")
      echo "Downloading ${config} step ${STEP}..."
      mkdir -p ~/hpc_checkpoints/${config}/peg_insertion_50demos/${STEP}/
      rsync -avz --progress \
        saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/${config}/peg_insertion_50demos/${STEP}/params/ \
        ~/hpc_checkpoints/${config}/peg_insertion_50demos/${STEP}/params/
    done

### Step 2: WSL2 -> Physical Ubuntu

Run from WSL2 terminal:

    scp -r ~/hpc_checkpoints/* \
      robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/

Password: (robolab-2's password)

---

## Transfer: RL Token Models

### Step 1: HPC -> WSL2

    mkdir -p ~/hpc_checkpoints/rl_token
    scp saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/checkpoints/rl_token/*.pt \
      ~/hpc_checkpoints/rl_token/

### Step 2: WSL2 -> Physical Ubuntu

    scp ~/hpc_checkpoints/rl_token/*.pt \
      robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/checkpoints/rl_token/

---

## Transfer: Git Repository Updates

    # On WSL2 (push to GitHub):
    cd ~/rlt_ur5e
    git add . && git commit -m "updates" && git push

    # On Physical Ubuntu (pull from GitHub):
    cd ~/ur5e_hande_workspace/rlt_ur5e
    git pull

    # On HPC (pull from GitHub):
    cd /data/beegfs/home/saifi/rlt_ur5e
    git pull

---

## Quick One-Liner: Transfer Single Model

    # Example: Transfer just pi0-FAST checkpoint
    # WSL2 terminal:
    scp -r saifi@hpc-headnode.iis.fhg.de:/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/pi0_fast_ur5e_peg_insertion_lora /tmp/ && \
    scp -r /tmp/pi0_fast_ur5e_peg_insertion_lora robolab-2@172.22.1.188:/home/robolab-2/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/

---

## Verify Transfer

On Physical Ubuntu after transfer:

    # Check checkpoints exist
    ls ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/*/peg_insertion_50demos/4999/params/

    # Check sizes (params should be ~5-7 GiB per model)
    du -sh ~/ur5e_hande_workspace/rlt_ur5e/openpi_ur5e/openpi-ur5e/checkpoints/*/

---

## Estimated Transfer Times

| What | Size | HPC->WSL2 | WSL2->Ubuntu |
|------|------|-----------|-------------|
| Full checkpoint (1 model) | ~12 GiB | ~5 min | ~2 min |
| Params only (1 model) | ~5 GiB | ~2 min | ~1 min |
| All 3 models (params only) | ~15 GiB | ~6 min | ~3 min |
| RL Token model | ~50 MB | instant | instant |
| Git repo | ~5 MB | instant | instant |
