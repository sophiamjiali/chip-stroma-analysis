# ==============================================================================
# Script:           logger.py
# Purpose:          Initializes logger output
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import logging
import sys
import wandb

from lightning.pytorch.loggers import WandbLogger

# =====| System Output Logger |=================================================

def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Create or retrieve a logger with safe default configuration. Enforces
    absolute determinism for logging and does not depend on upstream
    logging configuration.

    Ensures logging works in scripts and SLURM environments where
    no prior logging configuration exists.
    """

    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers = [logging.StreamHandler(sys.stdout)],
        force    = True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    return logger

# =====| Training Loggers |=====================================================

def configure_loggers(project: str,
                      group: str,
                      trial = None) -> WandbLogger:
    
    logger = setup_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("Step 05: Logging Configuration")
    logger.info(f"- Project: {project}")
    logger.info(f"- Group: {group}")
    logger.info("-" * 50)

    # Start a new run with this trial if a sweep is detected
    if trial is not None:
        logger.info("Sweep detected, initializing a new run logger within it")
        logger.info(f"Initialized a logger for sweep trial {trial.number}")
        run = wandb.init(
            project = project,
            group   = group,
            name    = f"trial_{trial.number}",
            id      = f"{group}_trial_{trial.number}",
            resume  = "never",
            config  = trial.params,
            reinit  = True
        )


    # Else, start a regular run outside of sweep configurations
    else:
        logger.info("Single run detected, initializing a standalone run logger")
        logger.info("Initialized a logger for train iteration as 'single_run'")
        run = wandb.init(
            project = project,
            group   = group,
            name    = "single_run",
            config  = {}
        )

    wandb_logger = WandbLogger(
        experiment = run,
        log_model  = False
    )

    logger.info("=" * 50)
    return wandb_logger

# [END]