#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/%x/%x_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/%x/%x_%j.err
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/chip-stroma-analysis/$1

# Activate the virtual environment
export LD_LIBRARY_PATH=/cluster/home/t111631uhn/miniconda3/lib:$LD_LIBRARY_PATH
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $1"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "=========================================="

CONFIG_DIR=/cluster/home/t144807uhn/chip-stroma-analysis/configs/hpc

srun python -u scripts/evaluate.py \
    --config_dir $CONFIG_DIR \
    --version v0

echo "=========================================="
echo "End: $(date)"
echo "=========================================="