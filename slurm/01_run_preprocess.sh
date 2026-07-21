#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/%x/%x_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/%x/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/chip-stroma-analysis/$1

# Activate the virtual environment
source /cluster/home/t144807uhn/envs/chip-stroma-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

# Avoid PyTorch Lightning SLURM mis-detection issues
unset SLURM_NTASKS
unset SLURM_JOB_NAME

export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=0

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $1"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "=========================================="

srun python scripts/01_preprocess.py \
    --config_dir configs

echo "=========================================="
echo "End: $(date)"
echo "=========================================="