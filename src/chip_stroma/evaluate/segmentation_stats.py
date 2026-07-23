# ==============================================================================
# Script:           segmentation_stats.py
# Purpose:          Statistics for segmentation model performance
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import optuna
import wandb

import pandas as pd
import numpy as np

from typing import cast

def per_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Computes macro Dice/precision/recall per fold, per patient."""

    # Compute per-patient nanmeans (patch -> patient)
    signal = predictions[predictions['has_signal']]
    per_patient = (
        signal.groupby(['fold', 'sample_id'])[['dice', 'precision', 'recall']]
        .mean()
        .reset_index()
    )

    # Compute fold-level macro mean, retaining per-patient rows
    fold_macro = (
        per_patient.groupby('fold')[['dice', 'precision', 'recall']]
        .mean()
        .reset_index()
        .assign(sample_id = 'MACRO')
    )

    return pd.concat([per_patient, fold_macro], ignore_index = True)
    

def optuna_importance(db_path: str) -> pd.DataFrame:
    """Wraps parameter importance call from Optuna, returns for plotting."""
    study = optuna.load_study(study_name = None, storage = db_path)
    importances = optuna.importance.get_param_importances(study)

    return pd.DataFrame({
        'param'     : list(importances.keys()), 
        'importance': list(importances.values())
    })


def threshold_sweep(probs: np.ndarray, 
                    gt: np.ndarray, 
                    thresholds = np.linspace(0.1, 0.9, 17)) -> pd.DataFrame:
    """
    Pooled precision/recall/dice at each threshold, justifying Otsu versus a 
    learned threshold for downstream fibroblast quantification.
    """

    probs, gt = probs.ravel(), gt.ravel().astype(bool)

    rows = []
    for t in thresholds:

        # Apply the threshold and compute each metric
        pred = probs >= t
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
 
        rows.append({
            "threshold": t, 
            "precision": precision, 
            "recall": recall, 
            "dice": dice
        })
 
    return pd.DataFrame(rows)


def select_overlay_cases(per_patient   : pd.DataFrame, 
                         n_per_category: int) -> pd.DataFrame:
    """
    Selects best/median/worst-Dice patients per fold for overlay QC figures.
    """

    cases = []
    for _, group in per_patient.groupby("fold"):
        ranked = (group.sort_values("dice", ascending = False)
                  .reset_index(drop = True))
        n = len(ranked)
 
        best  = ranked.iloc[:n_per_category].assign(category  = "best")
        worst = ranked.iloc[-n_per_category:].assign(category = "worst")
 
        mid    = n // 2
        half   = n_per_category // 2
        median = (ranked.iloc[max(mid - half, 0): max(mid - half, 0) + 
                              n_per_category].assign(category = "median"))
 
        cases.extend([best, worst, median])
 
    return pd.concat(cases, ignore_index = True)


def top_k_trials_table(db_path: str, k: int) -> pd.DataFrame:
    """Top-k completed trials by val/dice, with swept hyperparameter values."""

    # Extract all completed trials from the Optuna records
    study = optuna.load_study(study_name = None, storage = db_path)
    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
    ]

    # Sort by validation dice score, default provided by Optuna
    ranked = sorted(
        trials,
        key     = lambda t: cast(float, t.value),
        reverse = True,
    )[:k]

    rows = [{"trial_number": t.number, "val_dice": t.value, **t.params} 
            for t in ranked]
    return pd.DataFrame(rows)



def multiseed_summary_table(trial_results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean +/- SD across seeds for top-3 trials, keyed by hyperparameter 
    configurations.
    """

    metric_cols = ["dice", "precision", "recall"]
    summary = (
        trial_results.groupby("trial_number")[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def final_cv_summary_table(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD across the 5 folds for the fixed finalist configuration."""
    
    metric_cols = ["dice", "precision", "recall"]
    summary = cv_results[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    return summary.reset_index().rename(columns={"index": "metric"})

# [END]