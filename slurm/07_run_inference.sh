#!/bin/bash
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Activate the virtual environment
export LD_LIBRARY_PATH=/cluster/home/t111631uhn/miniconda3/lib:$LD_LIBRARY_PATH
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/chip-stroma-analysis

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

CONFIG_DIR=/cluster/home/t144807uhn/chip-stroma-analysis/configs

srun python -u scripts/07_inference.py \
    --config_dir $CONFIG_DIR \
    --version $1