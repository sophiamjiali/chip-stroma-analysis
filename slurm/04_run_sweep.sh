#!/bin/bash

echo "=========================================="
echo "Job Name:        04_run_sweep.sh"
echo "Start time:      $(date)"
echo "=========================================="

cd /cluster/home/t144807uhn/chip-stroma-analysis/slurm

# Initialize the study first
jid1=$(sbatch \
    --parsable \
    --job-name=$1 \
    04a_init_study.sh $1)

# Submit model trials in a SLURM array; at most six concurrent jobs
sbatch \
    --dependency=afterok:$jid1 \
    --array=0-17%6 \
    --job-name=$1 \
    04b_submit_sweep.sh $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="