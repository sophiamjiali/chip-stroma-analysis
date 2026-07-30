#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
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

python scripts/05b_aggregate_multiseed.py \
    --config_dir $CONFIG_DIR \
    --version $VERSION

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# [END]