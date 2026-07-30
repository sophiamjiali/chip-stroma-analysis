# ==============================================================================
# Script:           04a_sweep.py
# Purpose:          Initializes Optuna SQL database for sweep SLURM array task
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/27/2026
# ==============================================================================

import optuna

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pathlib import Path
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


def initialize_study(version         : str,
                     seed            : int,
                     n_startup_trials: int,
                     n_warmup_steps  : int,
                     studies_dir     : Path) -> optuna.Study: 
    """Initializes or loads an Optuna study."""

    # First detect if the database was already initialized
    storage_path = Path(f"{studies_dir}/{version}.log")
    storage_path.parent.mkdir(parents = True, exist_ok = True)


    if storage_path.exists(): 
        logger.info("Detected Optuna journal log, loading existing study")
    else:
        logger.info("Optuna journal log not found, initializing new study")

    
    sampler = TPESampler(
        seed             = seed,
        multivariate     = True,
        n_startup_trials = n_startup_trials
    )
    logger.info("Successfully initialized the TPESampler")
    
    
    pruner = MedianPruner(
        n_startup_trials = n_startup_trials,
        n_warmup_steps   = n_warmup_steps
    )
    logger.info("Successfully initialized the MedianPruner")

    storage = JournalStorage(JournalFileBackend(str(storage_path)))

    study = optuna.create_study(
        study_name     = version,
        storage        = storage,
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True
    )

    logger.info("Successfuly initialized the Optuna study for sweeping")
    return study


def load_study(version: str, study_dir: str):

    # Temporary fix for legacy .db
    if version == "v6":
        storage_path = Path(f"{study_dir}/{version}.db")
        study = optuna.load_study(
            study_name = version,
            storage = f"sqlite:///{storage_path}"
        )
        return study



    storage_path = Path(f"{study_dir}/{version}.log")

    storage = JournalStorage(JournalFileBackend(str(storage_path)))

    study = optuna.load_study(
        study_name = version,
        storage    = storage
    )

    return study

# [END]