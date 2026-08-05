#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

set -euo pipefail

# Extract command-line arguments for clarity
PROJECT_ROOT=$1
VERSION=$2

# Initialize the standardized environment
source ${PROJECT_ROOT}/.env
source ${PROJECT_ROOT}/slurm/00_setup_env.sh

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

srun --export=ALL python -u scripts/01_preprocess.py \
    --config_dir configs

# [END]