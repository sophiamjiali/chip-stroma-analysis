#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/chip-stroma-analysis//test_gpu.out
#SBATCH --error=/cluster/home/t144807uhn/logs/chip-stroma-analysis/test_gpu.err
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

nvidia-smi