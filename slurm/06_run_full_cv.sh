#!/bin/bash

N_FOLDS=5

echo "=========================================="
echo "Job Name:        05_run_multiseed.sh"
echo "Number of Folds: $N_FOLDS"
echo "Start time:      $(date)"
echo "=========================================="

cd /cluster/home/t144807uhn/chip-stroma-analysis/slurm

# Submit model trials in a SLURM array
ARRAY_JOBID=$(sbatch --parsable --array=0-$((N_FOLDS-1)) --job-name=$1 06a_submit_full_cv.sh $1)

# Aggregate trial summaries after all complete
sbatch --dependency=afterok:$ARRAY_JOBID --job-name=$1 06b_aggregate_full_cv.sh $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="