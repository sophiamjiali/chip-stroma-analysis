#!/bin/bash
TOP_K=2; N_SEEDS=3
cd /cluster/home/t144807uhn/chip-stroma-analysis/slurm

ARRAY_JOBID=$(sbatch --parsable --array=0-$((TOP_K*N_SEEDS-1)) --job-name=v6 05a_submit_multiseed.sh v6)
sbatch --dependency=afterok:$ARRAY_JOBID --job-name=v6 05b_aggregate_multiseed.sh v6