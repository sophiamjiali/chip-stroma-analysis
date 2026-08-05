# ==============================================================================
# Script:           10_quantify.yaml
# Purpose:          Quantify fibroblasts in aSMA stains
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/30/2026
# ==============================================================================

import argparse as ap
import pandas as pd

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.quantification.density_score import (
    quantify_fold,
    aggregate_scores,
    summarize_sensitivity
)

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Quantification",
        config_path    = Path(args.config_dir) / "10_quantify.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "10_quantify.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Load the validation fold as a dataset
    manifest = initialize_train_manifest(
        train_path = config.paths.metadata.train_manifest,
        patch_path = config.paths.metadata.patch_manifest
    )

    # Initialize version results directory 
    quantify_dir = Path(config.paths.results) / args.version / "quantify"
    qa_mask_dir  = quantify_dir / "masks"
    qa_mask_dir.mkdir(parents = True, exist_ok = True)

    # Perform a threshold sensitivity sweep above, below, and at the provided
    base_thresh  = float(config.quantify.vessel_threshold)
    delta        = float(config.quantify.vessel_threshold_delta)
    thresholds   = [base_thresh - delta, base_thresh, base_thresh + delta]

    # Loop over all folds for full-CV models (or only one fold)
    if args.single_model:
        logger.info("Performing quantification on a single model")
        logger.info(f"Detected fold: {config.quantify.n_folds}")
        folds = [config.quantify.n_folds]
    else:
        logger.info("Performing quantification on full cross-validated models")
        folds = list(range(config.quantify.n_folds))

    all_rows = []
    for fold in folds:
        logger.info(f"Quantifying fold {fold + 1} / {len(folds)}")

        # Perform quantification on the specified fold
        all_rows.extend(
            quantify_fold(
                fold         = fold,
                manifest     = manifest,
                thresholds   = thresholds,
                base_thresh  = base_thresh,
                mask_dir     = qa_mask_dir,
                paths        = config.paths,
                version      = args.version,
                single_model = args.single_model
            )
        )

    patch_scores  = pd.DataFrame(all_rows)
    sample_scores = aggregate_scores(patch_scores)
    sensitivity   = summarize_sensitivity(sample_scores)

    patch_scores.to_csv(quantify_dir / "patch_fibroblast_density.csv", 
                        index = False)
    sample_scores.to_csv(quantify_dir / "sample_fibrblast_density.csv", 
                         index = False)
    sensitivity.to_csv(quantify_dir / "vessel_threshold_sensitivity.csv", 
                       index = False)

    logger.info("fQuantification  complete for all fold(s).")
    log_footer()
    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Fibroblast quantification.")
    parser.add_argument("--config_dir",   type = str, default = "configs/")
    parser.add_argument("--version",      type = str)
    parser.add_argument("--single_model", action = "store_true")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]