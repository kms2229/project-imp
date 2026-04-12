#!/bin/bash
#SBATCH --job-name=qevc_vqa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-vqa-%j.out
#SBATCH --error=slurm-vqa-%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# QEVC Training on VQA-CP v2 — BU SCC
# ============================================================
# Usage:
#   sbatch qevc/scripts/train_vqa.sh
#   sbatch qevc/scripts/train_vqa.sh --export=N_SAMPLES=500,EPOCHS=5  # sanity check
# ============================================================

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load modules
module load python3/3.10.12
module load cuda/11.8

# Activate environment (adjust path for SCC)
source ~/qevc_env/bin/activate

# Navigate to project
cd ~/qevc || cd "$SLURM_SUBMIT_DIR" || exit 1

# Default arguments (can be overridden via --export)
N_SAMPLES=${N_SAMPLES:-""}
EPOCHS=${EPOCHS:-""}
LAMBDA=${LAMBDA:-""}
N_LAYERS=${N_LAYERS:-""}
CONFIG=${CONFIG:-"configs/default.yaml"}

# Build command
CMD="python -m qevc.scripts.train_qevc --dataset vqacp --config $CONFIG --device cuda"

if [ -n "$N_SAMPLES" ]; then
    CMD="$CMD --n-samples $N_SAMPLES"
fi
if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi
if [ -n "$LAMBDA" ]; then
    CMD="$CMD --lam $LAMBDA"
fi
if [ -n "$N_LAYERS" ]; then
    CMD="$CMD --n-layers $N_LAYERS"
fi

echo "Running: $CMD"
eval "$CMD"

echo "Job finished: $(date)"
