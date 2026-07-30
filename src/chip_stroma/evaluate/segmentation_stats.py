# ==============================================================================
# Script:           segmentation_stats.py
# Purpose:          Statistics for segmentation model performance
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import optuna
import h5py
import random

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
                    calib_sample_size: int = 1000000,
                    thresholds   : np.ndarray = np.linspace(0.1, 0.9, 17)
                    ) -> pd.DataFrame:
    """
    Pooled precision/recall/dice at each threshold, justifying Otsu versus a 
    learned threshold for downstream fibroblast quantification.
    """

    fold_range = [None] if single_model else range(n_folds)
    counts = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in thresholds}

    # Reservoir sample for calibration
    reservoir_probs, reservoir_gt = [], []
    n_seen = 0
    rng = random.Random(0)

    # Compute threshold sweep per fold
    for f in fold_range:
        fold_dir = inference_dir if f is None else inference_dir / f"fold_{f}"

        # Stream directlry to the h5py files
        with h5py.File(fold_dir / "val_arrays.h5", "r") as h5f:
            probs = cast(h5py.Dataset, h5f["probs"])
            gt    = cast(h5py.Dataset, h5f["gt"])

            n = probs.shape[0]

            # Process the predictions in chunks
            for start in range(0, n, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n)

                # De-quantize uint8 -> [0,1] float32 for this chunk only
                probs = probs[start:end].astype(np.float32) / 255.0
                gt    = gt[start:end]

                probs_flat, gt_flat = probs.ravel(), gt.ravel()

                for t in thresholds:
                    pred = probs_flat >= t
                    counts[t]["tp"] += int(np.logical_and(pred, gt_flat).sum())
                    counts[t]["fp"] += int(np.logical_and(pred, ~gt_flat).sum())
                    counts[t]["fn"] += int(np.logical_and(~pred, gt_flat).sum())
                    counts[t]["tn"] += int(np.logical_and(~pred, ~gt_flat).sum())

                # Reservoir sampling: uniformly subsample pixels across the full stream
                for p, g in zip(probs_flat, gt_flat):
                    n_seen += 1
                    if len(reservoir_probs) < calib_sample_size:
                        reservoir_probs.append(p); reservoir_gt.append(g)
                    else:
                        j = rng.randint(0, n_seen - 1)
                        if j < calib_sample_size:
                            reservoir_probs[j] = p; reservoir_gt[j] = g

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
        
    sweep = pd.DataFrame(rows)

    # Confusion counts at the best (argmax-dice) threshold only
    best_t = sweep_df.loc[sweep_df["dice"].idxmax(), "threshold"]
    confusion_counts = {**counts[best_t], "threshold": float(best_t)}

    calib_sample = {"probs": np.array(reservoir_probs), "gt": np.array(reservoir_gt, dtype=bool)}

    return sweep, confusion_counts, calib_sample




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


def compute_per_patient_vessel_area(inference_dir: Path, 
                                    n_folds: int, 
                                    single_model: bool) -> pd.DataFrame:
    """
    Per-patient GT vessel-area fraction: sum(vessel pixels) / sum(tissue pixels),
    pooled across that patient's patches. Joined with per_fold_metrics dice in
    the visualize step for the dice-vs-vessel-area plot.
    """
    fold_range = [None] if single_model else range(n_folds)
    rows = []

    for f in fold_range:
        fold_dir = inference_dir if f is None else inference_dir / f"fold_{f}"
        metrics = pd.read_csv(fold_dir / "patch_metrics.csv")
        with h5py.File(fold_dir / "val_arrays.h5", "r") as h5f:
            for sample_id, group in metrics.groupby("sample_id"):
                idxs = group.index.to_numpy()
                # gt already tissue-masked at inference time (vessel_masks_m)
                gt_chunk = h5f["gt"][idxs.min():idxs.max() + 1]
                vessel_px = gt_chunk.sum()
                tissue_px = gt_chunk.size  # denominator note below
                rows.append({"sample_id": sample_id, "fold": f if f is not None else 0,
                             "vessel_area_frac": vessel_px / tissue_px})

    return pd.DataFrame(rows)

# [END]