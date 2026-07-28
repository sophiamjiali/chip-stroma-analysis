#!/bin/bash
set -euo pipefail

TOP_K=2; N_SEEDS=3

# Extract command-line arguments for clarity
VERSION=$1

# Load the environment file
source "$(dirname "$0")/../.env"

echo "=========================================="
echo "Job Name:        05_run_multiseed.sh"
echo "Top K:           $TOP_K"
echo "Number of Seeds: $N_SEEDS"
echo "Start time:      $(date)"
echo "=========================================="

cd /cluster/home/t144807uhn/chip-stroma-analysis/slurm

# Submit model trials in a SLURM array
ARRAY_JOBID=$(sbatch \
    --parsable \
    --array=0-$((TOP_K*N_SEEDS-1)) \
    --job-name="${VERSION}_multiseed" \
    05a_submit_multiseed.sh \
    $PROJECT_ROOT \
    $VERSION)

# Aggregate trial summaries after all complete
sbatch \
    --dependency=afterok:$ARRAY_JOBID \
    --job-name="${VERSION}_multiseed" \
    05b_aggregate_multiseed.sh \
    $PROJECT_ROOT \
    $VERSION

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="