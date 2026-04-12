#!/bin/bash
#SBATCH --job-name=qevc_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8
#SBATCH --output=slurm-ablation-%A_%a.out
#SBATCH --error=slurm-ablation-%A_%a.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# QEVC Ablation Study — SLURM Array Job (BU SCC)
# ============================================================
# Runs 9 jobs in parallel:
#   Array 0-4: Lambda sweep  [0.1, 0.25, 0.5, 0.75, 1.0]
#   Array 5-8: Depth sweep   [2, 4, 6, 8]
#
# Usage:
#   sbatch qevc/scripts/ablation_array.sh
# ============================================================

echo "Job started: $(date)"
echo "Node: $(hostname) | Array Task ID: $SLURM_ARRAY_TASK_ID"

module load python3/3.10.12
module load cuda/11.8

source ~/qevc_env/bin/activate
cd ~/qevc || cd "$SLURM_SUBMIT_DIR" || exit 1

DATASET=${DATASET:-"vqacp"}
CONFIG=${CONFIG:-"configs/default.yaml"}

# Map array task ID to ablation parameter
LAMBDA_VALUES=(0.1 0.25 0.5 0.75 1.0)
DEPTH_VALUES=(2 4 6 8)

if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
    # Lambda sweep
    VALUE=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}
    echo "Running lambda ablation: lambda=$VALUE"
    python -m qevc.scripts.run_ablation \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --param lambda \
        --value "$VALUE"
else
    # Depth sweep
    IDX=$((SLURM_ARRAY_TASK_ID - 5))
    VALUE=${DEPTH_VALUES[$IDX]}
    echo "Running depth ablation: depth=$VALUE"
    python -m qevc.scripts.run_ablation \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --param depth \
        --value "$VALUE"
fi

echo "Job finished: $(date)"
