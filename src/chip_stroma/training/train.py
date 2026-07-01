# ==============================================================================
# Script:           train.py
# Purpose:          Exposes a reusable training function for runs and sweeps
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import logging

import lightning.pytorch as pl
import pandas as pd

from box import Box

from chip_stroma.utils.loggers import setup_logger, configure_loggers
from chip_stroma.utils.callbacks import configure_callbacks
from chip_stroma.models.model import VesselSegModule
from chip_stroma.data.datamodule import VesselSegmentationDataModule
from chip_stroma.models.model import build_model


logger = setup_logger(__name__)

# Suppress seed_everything automatic message (cosmetic only)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR) 


def train(manifest: pd.DataFrame,
          project: str,
          group: str,
          paths: Box,
          params: Box,
          seed: int,
          trial = None) -> dict:
    """
    Reusable training function for single runs and Optuna sweeps.

    Parameters
    ----------
    manifest   : Patch manifest of all patches to use for training
    project    : Global project identifier (e.g. chip_stroma_vessel_seg)
    group      : Unique project identifier (sweep name or train run name)
    paths      : Flat configurations for all relevant paths
    params     : Flat hyperparameters for model initialization
    seed       : Random seed.
    trial      : Optuna Trial, if called from a sweep. None for single runs.

    Returns
    -------
    Dict : trainer.callback_metrics at end of training.
    """

    logger.info("=" * 50)
    logger.info("Step 03: Train Initialization")
    logger.info(f"- Project: {project}")
    logger.info(f"- Group: {group}")
    logger.info(f"- Encoder Name: {params.model.encoder_name}")
    logger.info(f"- Encoder Weights: {params.model.encoder_weights}")
    logger.info("-" * 50)

    # Seed everything using the random state provided from configurations
    logger.info(f"Seeded all processes and workers with random state {seed}")
    pl.seed_everything(seed, workers = True)

    # Initialize the data module, model, and LightningModule
    logger.info("Successfully initialized the DataModule")
    datamodule = VesselSegmentationDataModule(
        manifest        = manifest,
        patch_dir       = paths.processed_data.patch_dir,
        vessel_mask_dir = paths.processed_data.vessel_mask_dir,
        tissue_mask_dir = paths.processed_data.tissue_mask_dir,
        fold            = params.data.fold,
        batch_size      = params.data.batch_size,
        num_workers     = params.data.num_workers,
        pos_prob        = params.data.pos_prob,
        seed            = params.data.seed
    )

    logger.info("Successfully initialized the Model")
    model = build_model(
        encoder_name    = params.model.encoder_name,
        encoder_weights = params.model.encoder_weights,
        in_channels     = params.model.in_channels,
        out_classes     = params.model.out_classes
    )

    logger.info("Successfully initialized the LightningModule")
    lit_module = VesselSegModule(
        model = model,
        lr           = params.module.lr,
        weight_decay = params.module.weight_decay,
        T_max        = params.trainer.max_epochs,
        ftl_alpha    = params.module.ftl_alpha,
        ftl_beta     = 1 - params.module.ftl_alpha,
        ftl_gamma    = params.module.ftl_gamma,
        ftl_weight   = params.module.ftl_weight,
        smooth       = params.module.smooth
    )

    logger.info("=" * 50)

    # Initialize all callbacks and loggers needed for training
    callbacks = configure_callbacks(
        trial                    = trial,
        early_stopping_patience  = params.trainer.early_stopping_patience,
        early_stopping_min_delta = params.trainer.early_stopping_min_delta,
    )

    ml_loggers = configure_loggers(
        project = project,
        group   = group,
        trial   = trial,
    )

    # Initialize the Trainer and fit the model
    trainer = pl.Trainer(
        max_epochs              = params.trainer.max_epochs,
        callbacks               = callbacks,
        logger                  = ml_loggers,
        accelerator             = "cpu",
        devices                 = 1,
        num_nodes               = 1,
        deterministic           = False,
        log_every_n_steps       = 1,
        gradient_clip_val       = params.trainer.gradient_clip_val,
        gradient_clip_algorithm = "norm",
        check_val_every_n_epoch = 1,
        enable_checkpointing    = False,
        enable_progress_bar     = False,
        fast_dev_run = True
    )

    logger.info("=" * 50)
    logger.info("Step 06: Model Training")
    logger.info(f"- Project: {project}")
    logger.info(f"- Group: {group}")
    logger.info("-" * 50)
    logger.info("Beginning model training")

    trainer.fit(model = lit_module, datamodule = datamodule)

    logger.info("Successfully completed model training")
    logger.info("=" * 50)
    
    return trainer.callback_metrics

# [END]