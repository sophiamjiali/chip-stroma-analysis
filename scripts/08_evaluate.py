# ==============================================================================
# Script:           08_evaluate.py
# Purpose:          Model evaluation on the validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import pandas as pd
import argparse as ap
import numpy as np

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import load_all_fold_patch_metrics

from chip_stroma.evaluate.segmentation_stats import (
    per_fold_metrics,
    optuna_importance,
    threshold_sweep,
    select_overlay_cases,
    top_k_trials_table,
    final_cv_summary_table
)

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Evaluation",
        config_path    = Path(args.config_dir) / "08_evaluate.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "08_evaluate.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )
    n_folds = int(config.evaluate.n_folds)

    # Initialize version results directory
    dst_dir = Path(config.paths.results) / args.version
    inference_dir = dst_dir / "inference"
    evaluate_dir  = dst_dir / "evaluate"
    evaluate_dir.mkdir(parents = True, exist_ok = True)

    # 1. Compute per-patient segmentation metrics, per fold
    logger.info(f"Successfully loaded patch predictions for {n_folds} folds")
    predictions = load_all_fold_patch_metrics(
        src_dir      = inference_dir,
        n_folds      = n_folds,
        single_model = args.single_model
    )

    fold_metrics = per_fold_metrics(predictions)
    fold_metrics.to_csv(evaluate_dir / "per_fold_metrics.csv", index = False)
    logger.info("Successfully computed and saved per-fold metrics")

    # 2. Compute a threshold sweep, pooled across folds
    thr        = config.evaluate.thresholds
    thresholds = np.linspace(thr[0], thr[1], thr[2])
    thresholds = threshold_sweep(
        inference_dir = inference_dir,
        n_folds       = n_folds,
        single_model  = args.single_model,
        thresholds    = thresholds
    )
    
    thresholds.to_csv(evaluate_dir / "threshold_sweep.csv", index = False)
    logger.info(f"Successfully conducted threshold sweep on {len(thresholds)} "
                f"candidates")

    # 3. Extract Optuna diagnostics from the sweep stage
    importance = optuna_importance(args.version, config.paths.studies)
    importance.to_csv(evaluate_dir / "optuna_importance.csv", index = False)
    logger.info("Successfully extracted Optuna importance diagnostics "
                "from sweep")

    # 4. Overlay case selection by best/median/worst Dice
    per_patient = (
        predictions[predictions['has_signal']]
        .groupby(['fold', 'sample_id'])[['dice', 'precision', 'recall']]
        .mean()
        .reset_index()
    )
    n_overlay_cases = config.evaluate.n_overlay_cases
    overlay_cases   = select_overlay_cases(per_patient, n_overlay_cases)
    overlay_cases.to_csv(evaluate_dir / "overlay_cases.csv", index = False)
    logger.info(f"Successfully selected {n_overlay_cases} candidate cases "
                f"for overlay")

    # 5. Compute summary tables
    top_k = top_k_trials_table(
        version   = args.version,
        study_dir = config.paths.studies,
        k         = config.evaluate.top_k_trials
    )
    top_k.to_csv(evaluate_dir / "top_k_trials.csv", index = False)
    logger.info("Successfully computed summary tables for top K trials")

    final_cv = pd.read_csv(dst_dir / "full_cv_summary.csv")
    final_cv = final_cv_summary_table(cv_results = final_cv)
    final_cv.to_csv(evaluate_dir / "final_cv_summary.csv", index = False)
    logger.info("Successfully computed the full CV summary atble")

    log_footer()

    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Model evaluation.")
    parser.add_argument("--config_dir",   type = str, default = "configs/")
    parser.add_argument("--version",      type = str)
    parser.add_argument("--single_model", action = "store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()

# [END]