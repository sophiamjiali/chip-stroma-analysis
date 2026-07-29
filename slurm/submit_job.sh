#!/bin/bash
set -euo pipefail
# ==============================================================================
# Script:           submit_job.sh
# Purpose:          Wrapper to submit a non-orchestrated workflow step
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/28/2026
#
# Note:             This wrapper is literally just so I can submit it prettier,
#                   allowing me to nest the logs and preserve a more informative
#                   job name to track running processes on the SLURM scheduler.
# ==============================================================================

STEP_NAME=${1:? "Usage: $0 <pipeline_step> <version>"}
VERSION=${2:? "Usage: $0 <pipeline_step> <version>"}

# ==============================================================================

# Load the environment file
source "$(dirname "$0")/../.env"

# Map pipeline step to its corresponding number
STEP_NUMBER=$(python - <<PY
import yaml

with open("${CONFIG_DIR}/pipeline.yaml") as f:
    steps = yaml.safe_load(f)["steps"]

print(steps["$STEP_NAME"])
PY
)

# Build the primary paths for job submission
LOG_DIR="${STEP_NAME}/${VERSION}"
SLURM_SCRIPT="${STEP_NUMBER}_run_${STEP_NAME}.sh"
SLURM_PATH="${PROJECT_ROOT}/slurm/${SLURM_SCRIPT}"

# Make the project-specific logs directory
mkdir -p $LOG_DIR

echo "=========================================="
echo "Pipeline Step: $STEP_NAME"
echo "Slurm Script : $SLURM_SCRIPT"
echo "Version      : $VERSION"
echo "Log Directory: $LOG_DIR"
echo "Start        : $(date)"
echo "=========================================="

# Pass in the project root as $1, and version as $2
sbatch \
    --job-name=${VERSION}_${STEP_NAME} \
    --output=${LOG_ROOT}/${LOG_DIR}/${VERSION}_%j.out \
    --error=${LOG_ROOT}/${LOG_DIR}/${VERSION}_%j.err \
    $SLURM_PATH \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End: $(date)"
echo "=========================================="