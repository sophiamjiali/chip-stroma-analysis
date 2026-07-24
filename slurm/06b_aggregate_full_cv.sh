#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/full_cv/%x/aggregate_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/full_cv/%x/aggregate_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/chip-stroma-analysis/full_cv/$1

# Activate the virtual environment
export LD_LIBRARY_PATH=/cluster/home/t111631uhn/miniconda3/lib:$LD_LIBRARY_PATH
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "=========================================="
echo "Job ID:             $SLURM_JOB_ID"
echo "Job Name:           $1"
echo "Node:               $SLURMD_NODENAME"
echo "Start time:         $(date)"
echo "=========================================="

python scripts/06b_aggregate_full_cv.py \
    --config_dir configs/ \
    --version $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="