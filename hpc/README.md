# HPC Setup for π0/π0.5 Fine-tuning

> Fine-tune π0 and π0.5 LoRA on V100 32GB GPUs (9 demos, peg insertion task)
> Cluster: Fraunhofer IIS HPC | GPUs: 4× Tesla V100 SXM2 32GB per node

---

## 🚀 Quick Start

### From HPC headnode:
```bash
ssh -x saifi@hpc-headnode.iis.fhg.de
cd /data/beegfs/home/saifi/
git clone git@github.com:saifi-shahrukh/rlt_ur5e.git
cd rlt_ur5e/hpc
bash 01_setup.sh          # one-time: install uv, create venv, wandb login
```

### From LOCAL machine (transfer dataset):
```bash
cd ~/ur5e_hande_workspace/rlt_ur5e/hpc   # or wherever repo is
bash 02_transfer_dataset.sh
```

### Back on HPC:
```bash
bash 03_train.sh both     # π0 + π0.5 in parallel
bash 04_status.sh         # monitor progress
# Watch live: https://wandb.ai → project 'rlt-ur5e'
```

### When done, from LOCAL:
```bash
bash 05_download_checkpoints.sh
```

---

## 📁 Scripts

| Script | Run On | Purpose |
|--------|--------|----------|
| `01_setup.sh` | HPC | Install uv, create venv, setup paths |
| `02_transfer_dataset.sh` | LOCAL | rsync 9-demo dataset to cluster |
| `03_train.sh` | HPC | Submit SLURM jobs (norm/pi0/pi05/both/all) |
| `04_status.sh` | HPC | Monitor jobs, logs, checkpoints |
| `05_download_checkpoints.sh` | LOCAL | Pull trained weights back |
| `06_interactive.sh` | HPC | Get interactive GPU shell |
| `slurm/norm_stats.sh` | (auto) | Compute norm stats for π0.5 |
| `slurm/pi0.sh` | (auto) | Train π0 LoRA (30k steps, ~2h) |
| `slurm/pi05.sh` | (auto) | Train π0.5 LoRA (30k steps, ~3h) |

---

## Key Details

- **Config names:** `pi0_ur5e_peg_insertion_lora` / `pi05_ur5e_peg_insertion_lora`
- **Dataset:** `saifi/ur5e-peg-insertion-dual` (9 demos, dual camera)
- **Norm stats:** All 3 configs pre-computed ✓ (pi0, pi0-FAST, pi0.5)
- **Base weights:** Downloaded from `gs://openpi-assets/checkpoints/` (auto by training script)
- **W&B:** Online logging — see live loss curves at https://wandb.ai
- **No containers needed** — uses `uv` venv directly

---

## Expected Times

| Step | GPU | Duration |
|------|-----|----------|
| Norm stats (π0.5) | 1× V100 | ~10 min |
| π0 LoRA (30k steps) | 1× V100 | ~2h |
| π0.5 LoRA (30k steps) | 1× V100 | ~3h |
