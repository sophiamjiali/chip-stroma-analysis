#!/bin/bash
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

set -euo pipefail

# Extract command-line arguments for clarity
PROJECT_ROOT=$1
VERSION=$2

# Initialize the standardized environment
source ${PROJECT_ROOT}/.env
source ${PROJECT_ROOT}/slurm/00_setup_env.sh

# Configure WandB tracking for offline only (compute nodes have no internet)
export WANDB_PROJECT="chip-stroma"
export WANDB_MODE=offline
export WANDB_DIR="/cluster/home/t144807uhn/logs/chip-stroma-analysis/wandb/multiseed/$1/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WANDB_DIR"

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

srun python scripts/05a_multiseed.py \
    --config_dir $CONFIG_DIR \
    --version $VERSION \
    --task_id $SLURM_ARRAY_TASK_ID

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# [END]