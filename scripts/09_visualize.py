# ==============================================================================
# Script:           09_visualize.yaml
# Purpose:          Generates overlays on representative patches
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import wandb
import optuna

import argparse as ap

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import load_overlay_arrays, load_csv_inputs

from chip_stroma.visualize.segmentation_plots import (
    plot_fold_boxplots,
    plot_training_curves,
    plot_pr_curve,
    plot_patient_dice_violin,
    plot_overlay_panel,
    plot_optuna_importance,
    plot_optuna_parallel_coords
)

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Visualization",
        config_path    = Path(args.config_dir) / "09_visualize.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "09_visualize.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Initialize the version figure directory
    evaluate_dir  = Path(config.paths.results) / args.version / "evaluate"
    inference_dir = Path(config.paths.results) / args.version / "inference"
    figure_dir    = Path(config.paths.figures) / args.version / "segmentation"
    figure_dir.mkdir(parents = True, exist_ok = True) 

    # 1. Per-fold macro metric distribution (excludes MACRO summary rows)
    per_fold = load_csv_inputs(evaluate_dir / "per_fold_metrics.csv")

    plot_fold_boxplots(
        per_fold  = per_fold[per_fold['sample_id'] != "MACRO"],
        metric    = config.visualize.fold_boxplot_metric,
        save_path = figure_dir / "fold_boxplots.png"
    )

    # 2. Training cures for the selected finalist run
    metrics     = config.visualize.training_curve_metrics
    run         = wandb.Api().run(config.visualize.training_curve_run)
    run_history = run.history(keys = ['epoch', *metrics])

    plot_training_curves(
        run_history = run_history,
        metrics     = tuple(metrics),
        save_path   = figure_dir / "training_curves.png"
    )

    # 3. Precision/recall vs. threshold; justify Otsu over fixed threshold
    threshold_sweep = load_csv_inputs(evaluate_dir / "threshold_sweep.csv")

    plot_pr_curve(
        threshold_sweep = threshold_sweep,
        save_path       = figure_dir / "pr_curve.png"
    )

    # 4. Per-patient Dice violin plots; highlight outlier patient
    per_patient = load_csv_inputs(evaluate_dir / "per_fold_metrics.csv")
    per_patient = per_patient[per_patient['sample_id'] != 'MACRO']

    plot_patient_dice_violin(
        per_patient       = per_patient,
        highlight_patient = config.visualize.highlight_patient,
        save_path         = figure_dir / "patient_dice_violin.png"
    )

    # 5. Overlay panels for best/median/worst QC cases selected by 08_evaluate
    overlay_cases = load_csv_inputs(evaluate_dir / "overlay_cases.csv")
    overlay_dir = figure_dir / "overlays"
    overlay_dir.mkdir(parents = True, exist_ok = True)

    for _, case in overlay_cases.iterrows():
        image, gt_mask, pred_mask = load_overlay_arrays(
            src_dir   = inference_dir,
            fold      = case['fold'],
            sample_id = case['sample_id']
        )

        plot_name = (f"{case['category']}_fold{case['fold']}_" + 
                     f"{case['sample_id']}.png")

        plot_overlay_panel(
            image, gt_mask, pred_mask,
            dice_score = case['dice'],
            save_path = overlay_dir / plot_name
        )

    # 6. Optuna diagnostics; fANOVA importance and parallel coordinates
    importance = load_csv_inputs(evaluate_dir / "optuna_importance.csv")
    plot_optuna_importance(
        importance,
        save_path = figure_dir / "optuna_importance.png"
    )

    db    = config.paths.studies / f"{args.version}.db"
    study = optuna.load_study(study_name = None, storage = f"sqlite:///{db}")

    plot_optuna_parallel_coords(
        study.trials_dataframe(),
        save_path = figure_dir / "optuna_parallel_coords.png"
    )

    log_footer()
    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Train a single run of the model.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version", type = str)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]