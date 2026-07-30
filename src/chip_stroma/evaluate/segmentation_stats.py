# ==============================================================================
# Script:           segmentation_stats.py
# Purpose:          Statistics for segmentation model performance
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import optuna
import h5py

import pandas as pd
import numpy as np

from typing import cast

from chip_stroma.training.create_study import load_study

CHUNK_SIZE = 512


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
    

def optuna_importance(version: str, study_dir: str) -> pd.DataFrame:
    """Wraps parameter importance call from Optuna, returns for plotting."""
    study       = load_study(version = version, study_dir = study_dir)
    importances = optuna.importance.get_param_importances(study)

    return pd.DataFrame({
        'param'     : list(importances.keys()), 
        'importance': list(importances.values())
    })


def threshold_sweep(inference_dir: pd.DataFrame,
                    n_folds      : int,
                    single_model : bool,
                    thresholds   : np.ndarray = np.linspace(0.1, 0.9, 17)
                    ) -> pd.DataFrame:
    """
    Pooled precision/recall/dice at each threshold, justifying Otsu versus a 
    learned threshold for downstream fibroblast quantification.
    """

    fold_range = [None] if single_model else range(n_folds)
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    # Compute threshold sweep per fold
    for f in fold_range:
        fold_dir = inference_dir if f is None else inference_dir / f"fold_{f}"

        # Stream directlry to the h5py files
        with h5py.File(fold_dir / "val_arrays.h5", "r") as h5f:
            probs = cast(h5py.Dataset, h5f["probs"])
            gt    = cast(h5py.Dataset, h5f["gt"])

            # Process the predictions in chunks
            for start in range(0, probs.shape[0], CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, probs.shape[0])

                # De-quantize uint8 -> [0,1] float32 for this chunk only
                probs = probs[start:end].astype(np.float32) / 255.0
                gt    = gt[start:end]

                probs_flat, gt_flat = probs.ravel(), gt.ravel()

                for t in thresholds:
                    pred = probs_flat >= t
                    counts[t]["tp"] += np.logical_and(pred, gt_flat).sum()
                    counts[t]["fp"] += np.logical_and(pred, ~gt_flat).sum()
                    counts[t]["fn"] += np.logical_and(~pred, gt_flat).sum()

    # Compute metrics from accumulated global counts (pooled sweep)
    rows = []
    for t in thresholds:
        tp, fp, fn = counts[t]["tp"], counts[t]["fp"], counts[t]["fn"]
        precision  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall     = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        dice       = (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 
                      else np.nan)
        
        rows.append({
            "threshold": t, 
            "precision": precision,
            "recall"   : recall, 
            "dice"     : dice
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


def top_k_trials_table(version: str, study_dir: str, k: int) -> pd.DataFrame:
    """Top-k completed trials by val/dice, with swept hyperparameter values."""

    # Extract all completed trials from the Optuna records
    study = load_study(version, study_dir)
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
        trial_results.groupby("trial_num")[metric_cols]
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