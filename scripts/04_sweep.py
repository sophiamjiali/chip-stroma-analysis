# ==============================================================================
# Script:           04_sweep.py
# Purpose:          WandB sweep agent entrypoint
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import torch
import optuna

import argparse as ap

from pathlib import Path
from functools import partial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.training.objective import objective
from chip_stroma.utils.callbacks import make_checkpoint_callback

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    pipeline_path = Path(args.config_dir) / "sweeps" / f"{args.version}.yaml"
    log_header(
        pipeline_stage = "Sweep",
        config_path    = pipeline_path,
        version        = args.version
    )
    
    # 1. Load workflow and path configurations; sweeps are nested in a folder
    config = load_configs(
        pipeline    = pipeline_path,
        paths       = Path(args.config_dir) / "00_paths.yaml",
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
    study_name = config.sweep.study.group
    storage = f"sqlite:///{config.paths.studies}/{study_name}.db"

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True
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

    # Initialize checkpointing callback for best trial
    checkpoint_callback = make_checkpoint_callback(
        checkpoint_dir = config.paths.checkpoints,
        group          = study_name
    )

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
        n_trials  = n_trials,
        timeout   = timeout,
        catch     = (RuntimeError,),
        callbacks = [checkpoint_callback]
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
    logger.info(f"- Best Trial: {study.best_trial.number}")
    logger.info(f"- Best Validation Dice Score: {study.best_trial.value:.4f}")
    logger.info("=" * 50)

    log_footer()

    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Begin a hyperparameter sweep.")
    parser.add_argument("--config_dir", type = str, default = "configs/sweeps/")
    parser.add_argument("--version", type = str, default = "v1")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]