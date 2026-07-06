# ==============================================================================
# Script:           sweep.yaml
# Purpose:          WandB sweep agent entrypoint
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import torch
import optuna

import argparse as ap

from box import Box
from pathlib import Path
from datetime import datetime
from functools import partial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.training.objective import objective

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    pipeline_path = Path(args.config_dir) / "sweeps" / f"{args.version}.yaml"
    log_header(config_path = pipeline_path,
               version     = args.version)
    
    # 1. Load workflow and path configurations; sweeps are nested in a folder
    config = load_configs(
        pipeline    = pipeline_path,
        paths       = Path(args.config_dir) / "paths.yaml",
        config_name = "sweep",
        frozen      = False
    )

    # 2. Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )

    logger.info("=" * 50)
    logger.info("Step 03: Study Creation")
    logger.info("- Optimization Direction: maximize")
    logger.info(f"- Study Name (Group): {config.sweep.study.group}")
    logger.info(f"- Startup Trials: {config.sweep.experiment.n_startup_trials}")
    logger.info(f"- Warmup Steps: {config.sweep.experiment.n_warmup_steps}")
    logger.info("-" * 50)

    # 3. Initialize the sampler and pruner for the trial
    logger.info("Successfully initialized the TPESampler")
    sampler = TPESampler(
        seed             = config.sweep.data.seed,
        multivariate     = True,
        n_startup_trials = config.sweep.experiment.n_startup_trials
    )
    
    logger.info("Successfully initialized the MedianPruner")
    pruner = MedianPruner(
        n_startup_trials = config.sweep.experiment.n_startup_trials,
        n_warmup_steps   = config.sweep.experiment.n_warmup_steps
    )

    logger.info("Successfuly initialized the Optuna study for sweeping")
    study = optuna.create_study(
        direction  = "maximize",
        sampler    = sampler,
        pruner     = pruner,
        study_name = config.sweep.study.group
    )

    logger.info("=" * 50)

    # If a valid timeout was provided, overwrite n_trials
    if config.sweep.experiment.timeout != -1:
        timeout = config.sweep.experiment.timeout * 3600
        n_trials = None
    else:
        timeout, n_trials = None, config.sweep.experiment.n_trials

    logger.info("=" * 50)
    logger.info("Step 04: Hyperparameter Sweep")
    logger.info(f"- Timeout: {timeout}")
    logger.info(f"- Trials: {n_trials}")
    logger.info("-" * 50)
    logger.info("Beginning hyperparameter sweep")

    torch.set_num_threads(1)

    study.optimize(
        partial(
            objective, 
            manifest = manifest,
            project  = config.sweep.study.project,
            group    = config.sweep.study.group,
            paths    = config.paths,
            params   = config.sweep,
            seed     = config.sweep.data.seed
        ),
        n_trials = n_trials,
        timeout  = timeout
    )

    n_trials         = len(study.trials)
    trials_completed = len(study.get_trials(states = [TrialState.COMPLETE]))
    trials_pruned    = len(study.get_trials(states = [TrialState.PRUNED]))
    trials_failed    = len(study.get_trials(states = [TrialState.FAIL]))

    logger.info("Successfully completed hyperparameter sweep")
    logger.info(f"- Total Trials: {n_trials}")
    logger.info(f"- Completed Trials: {trials_completed}")
    logger.info(f"- Pruned Trials: {trials_pruned}")
    logger.info(f"- Failed Trails: {trials_failed}")
    logger.info("=" * 50)

    log_footer(config.paths, trial = study.best_trial)

    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Begin a hyperparameter sweep.")
    parser.add_argument("--config_dir", type = str, default = "configs/sweeps/")
    parser.add_argument("--version", type = str, default = "v1")
    
    return parser.parse_args()


def log_header(config_path: Path, version: str):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Segmentation UNet Training - Sweep")
    logger.info(f"- Version: {version}")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


def log_footer(cfg: Box, trial: FrozenTrial):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Best Trial: {trial.number}")
    logger.info(f"- Best Validation Dice Score: {trial.value:.4f}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]