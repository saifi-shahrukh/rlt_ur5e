# Documentation Index

All documentation for the RLT UR5e project, organized by topic.

---

## Quick Start

| Document | Description |
|----------|-------------|
| [../PIPELINE_GUIDE.md](../PIPELINE_GUIDE.md) | Complete end-to-end pipeline (training -> inference -> RL) |
| [../README.md](../README.md) | Project overview and structure |

---

## HPC Training

| Document | Description |
|----------|-------------|
| [HYPERPARAMETER_GUIDE.md](HYPERPARAMETER_GUIDE.md) | V100 optimization, batch sizes, memory budgets |
| [TRAINING_DETAILS.md](TRAINING_DETAILS.md) | LoRA ranks, training curves, what happens during training |
| [NEW_TASK_GUIDE.md](NEW_TASK_GUIDE.md) | Step-by-step guide for training on a new task |
| [../hpc/README.md](../hpc/README.md) | HPC scripts overview and usage |
| [../hpc/HPC_SETUP_README.md](../hpc/HPC_SETUP_README.md) | Initial HPC environment setup |
| [../hpc/ADDING_NEW_DATASET.md](../hpc/ADDING_NEW_DATASET.md) | How to add new datasets to HPC |
| [../hpc/hpc_gpu.md](../hpc/hpc_gpu.md) | GPU cluster hardware details |

---

## System Setup

| Document | Description |
|----------|-------------|
| [cluster.md](cluster.md) | Fraunhofer IIS cluster access and configuration |
| [commands.md](commands.md) | Common commands reference |
| [openpi_ur5e.md](openpi_ur5e.md) | OpenPI framework setup for UR5e |
| [lerobot_ur5e.md](lerobot_ur5e.md) | LeRobot data collection setup |

---

## RL and Policy

| Document | Description |
|----------|-------------|
| [rlt_ur5e.md](rlt_ur5e.md) | RLT (RL Token) architecture and training |
| [hil_serl.md](hil_serl.md) | Human-in-the-Loop SERL setup |

---

## Transfer Path (Network Topology)

    HPC (hpc-headnode.iis.fhg.de)
        |
        | SSH/SCP (Fraunhofer internal network)
        v
    WSL2 Ubuntu (r10028, 172.20.234.194)
        |
        | SCP (local network 172.22.x.x)
        v
    Physical Ubuntu (robolab-2, 172.22.1.188)
        |
        | Direct USB/Ethernet
        v
    UR5e Robot + Cameras

---

## Archived / Reference

| Document | Description |
|----------|-------------|
| [../EXECUTION_PLAN.md](../EXECUTION_PLAN.md) | Original execution plan |
| [../RLT_NEXT_STEPS.md](../RLT_NEXT_STEPS.md) | RLT implementation roadmap |
| [../RLT_UR5e_SETUP_README.md](../RLT_UR5e_SETUP_README.md) | Full RLT-UR5e system setup |
| [../COMMANDS.md](../COMMANDS.md) | Legacy commands reference |
| [../update_on_readme.md](../update_on_readme.md) | Update notes |
| [../openpi_ur5e/HPC_TRAINING.md](../openpi_ur5e/HPC_TRAINING.md) | OpenPI HPC training notes |
| [../openpi_ur5e/PEG_INSERTION_TASK.md](../openpi_ur5e/PEG_INSERTION_TASK.md) | Peg insertion task details |
| [../openpi_ur5e/UR5e_OpenPI_PIPELINE_README.md](../openpi_ur5e/UR5e_OpenPI_PIPELINE_README.md) | OpenPI pipeline for UR5e |
