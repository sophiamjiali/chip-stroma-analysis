#!/bin/bash

# Activate the virtual environment
export LD_LIBRARY_PATH=/cluster/home/t111631uhn/miniconda3/lib:$LD_LIBRARY_PATH
source /cluster/home/t144807uhn/envs/chip-stroma-env-gpu/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd $PROJECT_ROOT

# Mask Albumentions from checking for updates (no internet)
export NO_ALBUMENTATIONS_UPDATE=1

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

export PYTORCH_ENABLE_MPS_FALLBACK=0

# Point to the pre-downloaded Resnet34 imagenet weights
export TORCH_HOME="$HOME/.cache/torch"