#!/bin/bash

TOP_K=2; N_SEEDS=3

echo "=========================================="
echo "Job Name:       05_run_multiseed.sh"
echo "Top K:          $TOP_K"
echo "Top K:          $N_SEEDS"
echo "Start time:     $(date)"
echo "=========================================="

cd /cluster/home/t144807uhn/chip-stroma-analysis/slurm

# Submit model trials in a SLURM array
ARRAY_JOBID=$(sbatch --parsable --array=0-$((TOP_K*N_SEEDS-1)) --job-name=$1 05a_submit_multiseed.sh $1)

# Aggregate trial summaries after all complete
sbatch --dependency=afterok:$ARRAY_JOBID --job-name=$1 05b_aggregate_multiseed.sh $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="