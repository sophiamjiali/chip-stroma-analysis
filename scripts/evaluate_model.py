# ==============================================================================
# Script:           evaluate_model.py
# Purpose:          Evaluates fidelity and performance of UNet model
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/09/2026
#
# Notes:            Model checkpoint provided as `--ckpt_path`. Outputs to 
#                   `--dst_dir`
# ==============================================================================

import json
import torch

import pandas as pd
import numpy as np
import argparse as ap

from pathlib import Path
from datetime import datetime
from torchmetrics.classification import BinaryF1Score

from chip_stroma.models.model import VesselSegModule
from chip_stroma.data.dataset import VesselPatchDataset
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.config import load_configs

logger = setup_logger(__name__)


def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "evaluate.yaml")

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "evaluate.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )
    
    # Initialize the output directory for evaluation results
    dst_dir = Path(config.paths.results.evaluation)
    dst_dir.mkdir(parents = True, exist_ok = True)

    # Load the model and dataloader as specified in the configurations
    checkpoint_path = Path(config.evaluate.checkpoint_path)
    model           = VesselSegModule.load_from_checkpoint(
        checkpoint_path = checkpoint_path,
        map_location    = args.device
    )
    dataloader      = build_eval_dataloader()

    # Compute patch-evel metrics
    patch_metrics = compute_patch_metrics(model, dataloader, args.device)
    patch_metrics.to_csv(dst_dir / "patch_metrics.csv", index = False)

    # Compute sample/patient-level aggregates
    sample_metrics = patch_metrics.groupby('sample_id')[
        ['dice', 'precision', 'recall']].agg(['mean', 'std'])
    sample_metrics.to_csv(dst_dir / "sample_metrics.csv", index = False)

    # Compute threshold sensitivity
    threshold = threshold_sweep(model, dataloader, device)
    threshold.to_csv(dst_dir / "threshold_sensitivity.csv", index = False)
    
    # Compute overlays of inference on top of patches
    save_overlays(model, dataloader, device, patch_metrics, dst_dir)

    # Calibration check
    save_calibration_plot(patch_metrics, dst_dir)

    # Save a summary as a json
    summary = {
        "n_patches"     : len(patch_metrics),
        "n_patients"    : patch_metrics["patient_id"].nunique(),
        "mean_dice"     : patch_metrics["dice"].mean(skipna=True),
        "std_dice"      : patch_metrics["dice"].std(skipna=True),
        "mean_precision": patch_metrics["precision"].mean(skipna=True),
        "mean_recall"   : patch_metrics["recall"].mean(skipna=True),
        "best_threshold": threshold.loc[threshold["mean_dice"].idxmax(), 
                                        "threshold"],
    }
    with open(dst_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
 
    print(json.dumps(summary, indent=2))





# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Train a single run of the model.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--device", type = str)
    parser.add_argument("--name", type = str, default = "test")
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: UNet Model Evaluation")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

def log_footer(cfg, metrics):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")

    # ...
    
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]