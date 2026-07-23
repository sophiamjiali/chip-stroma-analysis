# ==============================================================================
# Script:           evaluate.py
# Purpose:          Model evaluation on the validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import pandas as pd
import argparse as ap

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import load_all_fold_patch_metrics

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Evaluation",
        config_path    = Path(args.config_dir) / "07_evaluate.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "07_evaluate.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )
    n_folds = int(config.evaluate.n_folds)

    # Initialize version results directory
    dst_dir = Path(config.paths.results) / args.version
    inference_dir = dst_dir / "inference"
    evaluate_dir  = dst_dir / "evaluate"
    evaluate_dir.mkdir(parents = True, exist_ok = True)

    # 1. Compute per-patient segmentation metrics, per fold
    predictions = load_all_fold_patch_metrics(
        src_dir = inference_dir,
        n_folds = n_folds
    )

    fold_metrics = per_fold_metrics(predictions)
    fold_metrics.to_csv(evaluate_dir / "per_fold_metrics.csv", index = False)

    # 2. Compute a threshold sweep, pooled across folds
    probs =pd.concat([pd.read_pickle(inference_dir/f"fold_{f}"/"val_probs.pkl") 
                      for f in range(n_folds)])
    gt    = pd.concat([pd.read_pickle(inference_dir / f"fold_{f}"/"val_gt.pkl") 
                      for f in range(n_folds)])
    
    thresholds = threshold_sweep(probs, gt)
    thresholds.to_csv(evaluate_dir / "threshold_sweep.csv", index = False)

    # 3. Extract Optuna diagnostics from the sweep stage
    database = config.paths.studies / f"{args.version}.db"
    importance = optuna_importance(str(database))
    importance.to_csv(evaluate_dir / "optuna_importance.csv", index = False)

    # 4. Boundary loss ablation
    bl_ablation = boundary_loss_ablation_curves(
        run_with_bl    = config.evaluate.bl_ablation_run_with,
        run_without_bl = config.evaluate.bl_ablation_run_without,
        ramp_epochs    = config.evaluate.bl_ramp_epochs,
    )
    bl_ablation.to_csv(evaluate_dir / "boundary_loss_ablation.csv", index=False)

    # 5. Overlay case selection by best/median/worst Dice
    per_patient = (
        predictions[predictions['has_signal']]
        .groupby(['fold', 'sample_id'])[['dice', 'precision', 'recall']]
        .mean()
        .reset_index()
    )

    overlay_cases = select_overlay_cases(
        per_patient, 
        n_per_category = config.evaluate_n_overlay_cases
    )
    overlay_cases.to_csv(evaluate_dir / "overlay_cases.csv", index = False)

    # 6. Compute summary tables
    top_k = top_k_trials_table(str(database), k = config.evaluate.top_k_trials)
    top_k.to_csv(evaluate_dir / "top_k_trials.csv", index = False)

    multiseed = pd.read_csv(dst_dir / "multiseed_summary.csv")
    multiseed = multiseed_summary_table(multiseed)
    multiseed.to_csv(evaluate_dir / "multiseed_summary.csv", index = False)

    final_cv = pd.read_csv(dst_dir / "full_cv_summary.csv")
    final_cv = final_cv_summary_table(final_cv)
    final_cv.to_csv(evaluate_dir / "final_cv_summary.csv", index = False)

    log_footer()

    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Model evaluation.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version", type = str)
    
    return parser.parse_args()


if __name__ == "__main__":
    main()

# [END]