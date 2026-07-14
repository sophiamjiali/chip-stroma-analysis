# ==============================================================================
# Script:           objective.py
# Purpose:          Optuna objective function for hyperparameter sweep
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/25/2026
#
# Notes:            Objective maximises val/dice (global aggregation) as the
#                   primary metric; consistent with Isensee et al. (2021).
#                   Fold is fixed (fold 0) during sweeps; hyperparameters are
#                   tuned on a single fold then applied across all folds in the
#                   final run (Raschka, 2018; Cawley & Talbot, 2010).
#
#                   Sweep ranges provided in the configuration YAML file can be #                   overwritten by constants to hold constant in the sweep.
# ==============================================================================

import optuna
import torch

import pandas as pd

from box import Box
from optuna import Trial
from datetime import datetime
from copy import deepcopy

from chip_stroma.training.train import train
from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


def objective(trial: Trial,
              manifest: pd.DataFrame,
              project:  str,
              group:    str,
              paths:    Box,
              params:   Box, 
              seed:     int) -> float:
    """
        Optuna objective for vessel segmentation hyperparameter sweep.
 
    Suggests values for all sweep parameters from ranges defined in
    sweep.yaml, then delegates to train(). Returns val/dice for
    maximisation.
 
    Parameters
    ----------
    trial    : Optuna Trial object for parameter suggestion.
    manifest : Patch manifest (pre-filtered, fold-assigned).
    project  : W&B project identifier.
    group    : Sweep name — used as W&B group for trial grouping.
    paths    : Flat path config Box.
    params   : Full hyperparameter config Box (sweep ranges pre-loaded).
    seed     : Global random seed.
 
    Returns
    -------
    float : val/dice at end of trial (higher = better).
    """

    logger.info("=" * 50)
    logger.info("Step 05: Hyperparameter Suggestion")
    logger.info(f"- Trial: {trial.number}")
    logger.info("-" * 50)

    # Suggest hyperparameters from the provided sweep configuration ranges
    trial_params = deepcopy(params)
    trial_params = suggest_optimizer(params  = trial_params, trial = trial)
    trial_params = suggest_focal_loss(params = trial_params, trial = trial)
    trial_params = suggest_sampler(params    = trial_params, trial = trial)
    trial_params = suggest_trainer(params    = trial_params, trial = trial)

    logger.info("=" * 50)

    # Train the model with the suggested parameters
    metrics = train(
        manifest = manifest,
        project  = project,
        group    = group,
        paths    = paths,
        params   = trial_params,
        seed     = seed,
        trial    = trial
    )

    logger.info("~" * 50)
    logger.info(f"Project {project} - Group {group}: "
                f"trial {trial.number} results")
    logger.info(f"- Validation Loss: {metrics.get('val/loss')}")
    logger.info(f"- Validation Dice Score: {metrics.get('val/dice')}")
    logger.info(f"- Validation IoU Score: {metrics.get('val/iou')}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("~" * 50)

    # Return the validation Dice score as the key metric
    # Guard against NaN/pruned trials polluting best-trial selection
    val_dice = metrics.get('val/dice', float('nan'))
    if not torch.isfinite(torch.tensor(val_dice)):
        raise optuna.TrialPruned()
    
    return float(val_dice)


# =====| Helpers |==============================================================

def suggest_optimizer(params: Box, trial: Trial) -> Box:
    """Provides in-place suggestions for optimizer parameters."""

    params.module.lr = (
        trial.suggest_float("lr", *params.module.lr, log = True)
        if isinstance(params.module.lr, (list, tuple))
        else params.module.lr
    )

    params.module.weight_decay = (
        trial.suggest_float("weight_decay",*params.module.weight_decay,log=True)
        if isinstance(params.module.weight_decay, (list, tuple))
        else params.module.weight_decay
    )

    logger.info("Successfully suggested learning rate and weight decay "
                "values for Optimizer")
    return params


def suggest_focal_loss(params: Box, trial: Trial) -> Box:
    """Provides in-place suggestions for Focal Tversky Loss parameters."""

    # Derive the beta term from the suggested alpha value
    params.module.ftl_alpha = (
        trial.suggest_float("ftl_alpha", *params.module.ftl_alpha)
        if isinstance(params.module.ftl_alpha, (list, tuple))
        else params.module.ftl_alpha
    )
    params.module.ftl_beta = 1.0 - params.module.ftl_alpha

    params.module.ftl_gamma = (
        trial.suggest_float("ftl_gamma", *params.module.ftl_gamma)
        if isinstance(params.module.ftl_gamma, (list, tuple))
        else params.module.ftl_gamma
    )
    
    params.module.ftl_weight = (
        trial.suggest_float("ftl_weight", *params.module.ftl_weight)
        if isinstance(params.module.ftl_weight, (list, tuple))
        else params.module.ftl_weight
    )

    logger.info("Successfully suggested alpha, beta, gamma, and weight values "
                "for Focal Tversky Loss")
    return params

def suggest_boundary_loss(params: Box, trial: Trial) -> Box:
    """Provides in-place suggestions for Boundary Loss parameters."""
    
    params.module.bl_weight_target = (
        trial.suggest_float("bl_weight_target", *params.module.bl_weight_target)
        if isinstance(params.module.bl_weight_target, (list, tuple))
        else params.module.bl_weight_target
    )

    params.module.bl_ramp_epochs = (
        trial.suggest_int("bl_ramp_epochs", *params.module.bl_ramp_epochs)
        if isinstance(params.module.bl_ramp_epochs, (list, tuple))
        else params.module.bl_ramp_epochs
    )

    logger.info("Successfully suggested weight target and ramping epochs "
                "values for Boundary Loss")
    return params


def suggest_sampler(params: Box, trial: Trial) -> Box:
    """Provides in-place suggestions for the sampler."""

    params.data.pos_prob = (
        trial.suggest_float("pos_prob", *params.data.pos_prob)
        if isinstance(params.data.pos_prob, (list, tuple))
        else params.data.pos_prob
    )

    logger.info("Successfully suggested positive probability for "
                "Sampler")
    return params


def suggest_trainer(params: Box, trial: Trial) -> Box:
    """Provides in-place suggestions for the Trainer."""

    # Suggest linear-uniform for gradient clip value
    params.trainer.gradient_clip_val = (
        trial.suggest_float("gradient_clip_val", *params.trainer.gradient_clip_val)
        if isinstance(params.trainer.gradient_clip_val, (list, tuple))
        else params.trainer.gradient_clip_val
    )

    logger.info("Successfulyl suggested gradient clip value for Trainer")
    return params

# [END]