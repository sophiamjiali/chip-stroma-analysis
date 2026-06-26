#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/%x/%x_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/%x/%x_%j.err
#SBATCH --time=24:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/$1

# Activate the virtual environment
module load python3/3.12.11
source /cluster/home/t144807uhn/envs/chip-stroma-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "=========================================="
echo "Mini Sweep Job ID:  $SLURM_JOB_ID"
echo "Job Name:           $1"
echo "Node:               $SLURMD_NODENAME"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "Start time:         $(date)"
echo "=========================================="

export WANDB_PROJECT="chip-stroma"
export WANDB_MODE=offline
export WANDB_DIR="/cluster/home/t144807uhn/logs/chip-stroma-analysis/wandb/sweep/$1"
mkdir -p "$WANDB_DIR"

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=0

CONFIG_DIR=/cluster/home/t144807uhn/chip-stroma-analysis/configs/hpc

srun python scripts/sweep.py \
    --config_dir $CONFIG_PATH \
    --version $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="