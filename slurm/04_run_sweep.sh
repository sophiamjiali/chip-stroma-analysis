#!/bin/bash
set -euo pipefail
# ==============================================================================
# Script:           04_run_sweep.sh
# Purpose:          Wrapper to submit the sweep orchestrated step
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/29/2026
# ==============================================================================

STEP_NAME="sweep"
VERSION=$1

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
LOG_DIR="${LOG_ROOT}/${STEP_NUMBER}_${STEP_NAME}/${VERSION}"

# Make the project-specific logs directory
mkdir -p $LOG_DIR

echo "=========================================="
echo "Job Name   : 04_run_sweep.sh"
echo "Version    : $VERSION"
echo "Start time : $(date)"
echo "=========================================="

cd ${PROJECT_ROOT}/slurm

# Initialize the study first
jid1=$(sbatch \
    --parsable \
    --job-name="${VERSION}_sweep" \
    --output=${LOG_DIR}/init_study_%A.out \
    --error=${LOG_DIR}/init_study_%A.err \
    04a_init_study.sh \
    $PROJECT_ROOT \
    $VERSION)

# Submit model trials in a SLURM array; at most six concurrent jobs
sbatch \
    --dependency=afterok:$jid1 \
    --array=0-17%6 \
    --job-name="${VERSION}_sweep" \
    --output=${LOG_DIR}/sweep_%a_%A.out \
    --error=${LOG_DIR}/sweep_%a_%A.err \
    04b_submit_sweep.sh \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="