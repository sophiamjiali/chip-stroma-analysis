#!/bin/bash
set -euo pipefail

N_FOLDS=5

# Extract command-line arguments for clarity
VERSION=$1

# Load the environment file
source "$(dirname "$0")/../.env"

echo "=========================================="
echo "Job Name:        06_run_full_cv.sh"
echo "Number of Folds: $N_FOLDS"
echo "Start time:      $(date)"
echo "=========================================="

cd /cluster/home/t144807uhn/chip-stroma-analysis/slurm

# Submit model trials in a SLURM array
ARRAY_JOBID=$(sbatch \
    --parsable \
    --array=0-$((N_FOLDS-1)) \
    --job-name="${VERSION}_full_cv" \
    06a_submit_full_cv.sh \
    $PROJECT_ROOT \
    $VERSION)

# Aggregate trial summaries after all complete
sbatch \
    --dependency=afterok:$ARRAY_JOBID \
    --job-name="${VERSION}_full_cv" \
    06b_aggregate_full_cv.sh \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="