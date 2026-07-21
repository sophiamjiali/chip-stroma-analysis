#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/%x/%x_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/%x/%x_%j.err
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/$1
mkdir -p /cluster/home/t144807uhn/chip-stroma-analysis/studies

# Activate the virtual environment
export LD_LIBRARY_PATH=/cluster/home/t111631uhn/miniconda3/lib:$LD_LIBRARY_PATH
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "=========================================="
echo "Mini Sweep Job ID:  $SLURM_JOB_ID"
echo "Job Name:           $1"
echo "Node:               $SLURMD_NODENAME"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "Start time:         $(date)"
echo "=========================================="

# Configure WandB tracking for offline only (compute nodes have no internet)
export WANDB_PROJECT="chip-stroma"
export WANDB_MODE=offline
export WANDB_DIR="/cluster/home/t144807uhn/logs/chip-stroma-analysis/wandb/sweep/$1"
mkdir -p "$WANDB_DIR"

# Mask Albumentions from checking for updates (no internet)
export NO_ALBUMENTATIONS_UPDATE=1

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

export PYTORCH_ENABLE_MPS_FALLBACK=0

# Point to the pre-downloaded Resnet34 imagenet weights
export TORCH_HOME="$HOME/.cache/torch"

CONFIG_DIR=/cluster/home/t144807uhn/chip-stroma-analysis/configs

srun python scripts/sweep.py \
    --config_dir $CONFIG_DIR \
    --version $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="