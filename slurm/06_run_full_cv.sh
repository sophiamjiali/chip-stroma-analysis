#!/bin/bash
set -euo pipefail
# ==============================================================================
# Script:           06_run_inference.sh
# Purpose:          Wrapper to submit the full CV orchestrated step
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/28/2026
# ==============================================================================

N_FOLDS=5

STEP_NAME="full_cv"
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
LOG_DIR="${STEP_NAME}/${VERSION}"

# Make the project-specific logs directory
mkdir -p $LOG_DIR

echo "=========================================="
echo "Job Name:        06_run_full_cv.sh"
echo "Number of Folds: $N_FOLDS"
echo "Start time:      $(date)"
echo "=========================================="

cd ${PROJECT_ROOT}/slurm

# Submit model trials in a SLURM array
ARRAY_JOBID=$(sbatch \
    --parsable \
    --array=0-$((N_FOLDS-1)) \
    --job-name="${VERSION}_full_cv" \
    --output=${LOG_ROOT}/${LOG_DIR}/fold_%a_%A.out \
    --error=${LOG_ROOT}/${LOG_DIR}/fold_%a_%A.err \
    06a_submit_full_cv.sh \
    $PROJECT_ROOT \
    $VERSION)

# Aggregate trial summaries after all complete
sbatch \
    --dependency=afterok:$ARRAY_JOBID \
    --job-name="${VERSION}_full_cv" \
    --output=${LOG_ROOT}/${LOG_DIR}/aggregate_%A.out \
    06b_aggregate_full_cv.sh \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="