#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/multiseed/%x/aggregate_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/multiseed/%x/aggregate_%j.err
#SBATCH --account=kumargroup_gpu
#SBATCH -p cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate
cd /cluster/home/t144807uhn/chip-stroma-analysis
python scripts/05b_aggregate_multiseed.py --config_dir configs/ --version $1