#!/bin/bash
# ==============================================================================
# Script:           07_submit_inference.sh
# Purpose:          Wrapper to submit the inference workflow step
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/22/2026
#
# Note:             This wrapper is literally just so I can submit it prettier,
#                   allowing me to nest the logs and preserve a more informative
#                   job name to track running processes on the SLURM scheduler.
# ==============================================================================

set -euo pipefail

# Extract the project root from the environment file
source "$(dirname "$0")/../.env"

# Build the primary paths for job submission
VERSION=${1:? "Usage: $0 <version>"}
LOG_DIR=inference/$VERSION
SLURM_SCRIPT="07_run_inference.sh"
SLURM_PATH=$PROJECT_ROOT/slurm/$SLURM_SCRIPT

# Make the project-specific logs directory
mkdir -p $LOG_DIR

echo "=========================================="
echo "Pipeline Step: Inference"
echo "Slurm Script : $SLURM_SCRIPT"
echo "Version      : $VERSION"
echo "Log Directory: $LOG_DIR"
echo "Start        : $(date)"
echo "=========================================="

sbatch \
    --job-name=${VERSION}_inference \
    --output=${LOG_ROOT}/${LOG_DIR}/${VERSION}_%j.out \
    --error=${LOG_ROOT}/${LOG_DIR}/${VERSION}_%j.err \
    $SLURM_PATH \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End: $(date)"
echo "=========================================="