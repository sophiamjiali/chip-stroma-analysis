# ==============================================================================
# Script:           segmentation_stats.py
# Purpose:          Statistics for segmentation model performance
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import pandas as pd
import numpy as np

def per_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Computes macro Dice/precision/recall per fold, per patient."""

    # Aggregate confusion counts


def optuna_importance(db_path: str) -> pd.DataFrame:
    """Wraps parameter importance call from Optuna, returns for plotting."""



def threshold_sweep(probs: np.ndarray, 
                    gt: np.ndarray, 
                    thresholds = np.linspace(0.1, 0.9, 17)) -> pd.DataFrame:
    """
    Precision/recall/dice at each threshold, justifying Otsu versus a learned 
    threshold for fibroblast quantification.
    """


def multiseed_summary(trial_results: list[dict]) -> pd.DataFrame:
    """
    Mean +/- SD across seeds for top-3 trials, keyed by hyperparameter 
    configurations.
    """

# [END]