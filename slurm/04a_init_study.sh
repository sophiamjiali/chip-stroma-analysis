#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/%x/init_study_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/%x/init_study_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/chip-stroma-analysis/sweep/$1
mkdir -p /cluster/home/t144807uhn/chip-stroma-analysis/studies

# Activate the virtual environment
export LD_LIBRARY_PATH=/cluster/home/t111631uhn/miniconda3/lib:$LD_LIBRARY_PATH
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "=========================================="
echo "Sweep Job ID:       $SLURM_JOB_ID"
echo "Job Name:           $1"
echo "Node:               $SLURMD_NODENAME"
echo "Start time:         $(date)"
echo "=========================================="

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

export PYTORCH_ENABLE_MPS_FALLBACK=0

CONFIG_DIR=/cluster/home/t144807uhn/chip-stroma-analysis/configs

srun python scripts/04a_init_study.py \
    --config_dir $CONFIG_DIR \
    --version $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="