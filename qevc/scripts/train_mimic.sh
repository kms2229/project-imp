#!/bin/bash
#SBATCH --job-name=qevc_mimic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-mimic-%j.out
#SBATCH --error=slurm-mimic-%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# QEVC Training on MIMIC-III — BU SCC
# ============================================================
# Usage:
#   sbatch qevc/scripts/train_mimic.sh
# ============================================================

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module load python3/3.10.12
module load cuda/11.8

source ~/qevc_env/bin/activate
cd ~/qevc || cd "$SLURM_SUBMIT_DIR" || exit 1

CONFIG=${CONFIG:-"configs/default.yaml"}

python -m qevc.scripts.train_qevc \
    --dataset mimic \
    --config "$CONFIG" \
    --device cuda

echo "Job finished: $(date)"
