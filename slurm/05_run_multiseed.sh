#!/bin/bash
set -euo pipefail

# ==============================================================================
# Script:           05_run_multiseed.sh
# Purpose:          Wrapper to submit the multiseed orchestrated step
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/28/2026
# ==============================================================================

TOP_K=2
N_SEEDS=3

STEP_NAME="multiseed"
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
LOG_DIR="${LOG_ROOT}/${STEP_NAME}/${VERSION}"

# Make the project-specific logs directory
mkdir -p $LOG_DIR

echo "=========================================="
echo "Job Name:        05_run_multiseed.sh"
echo "Top K:           $TOP_K"
echo "Number of Seeds: $N_SEEDS"
echo "Start time:      $(date)"
echo "=========================================="

cd ${PROJECT_ROOT}/slurm

# Submit model trials in a SLURM array
ARRAY_JOBID=$(sbatch \
    --parsable \
    --array=0-$((TOP_K*N_SEEDS-1)) \
    --job-name="${VERSION}_multiseed" \
    --output=${LOG_DIR}/combo_%a_%A.out \
    --error=${LOG_DIR}/combo_%a_%A.err \
    05a_submit_multiseed.sh \
    $PROJECT_ROOT \
    $VERSION)

# Aggregate trial summaries after all complete
sbatch \
    --dependency=afterok:$ARRAY_JOBID \
    --job-name="${VERSION}_multiseed" \
    --output=${LOG_DIR}/aggregate_%A.out \
    --error=${LOG_DIR}/aggregate_%A.err \
    05b_aggregate_multiseed.sh \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="