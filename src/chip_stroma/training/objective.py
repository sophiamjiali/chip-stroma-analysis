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
#                   alpha + beta = 1 constraint (Abraham & Khan, 2019) is
#                   enforced by deriving ftl_beta from ftl_alpha.
# ==============================================================================

import pandas as pd

from box import Box
from optuna import Trial

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

# [END]