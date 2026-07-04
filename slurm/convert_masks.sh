#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis/convert_masks.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/convert_masks.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Activate the virtual environment
module load python3/3.12.11
#source /cluster/home/t144807uhn/envs/chip-stroma-env/bin/activate
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $1"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "=========================================="

srun python scripts/convert_masks.py \
    /cluster/projects/kumargroup/ch_fibrosis_analysis/trident/masks \
    /cluster/projects/kumargroup/sophia/chip-stroma-analysis/data/vessel_masks

echo "=========================================="
echo "Exit Code: $?"
echo "End: $(date)"
echo "=========================================="