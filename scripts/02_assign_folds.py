# ==============================================================================
# Script:           assign_folds.py
# Purpose:          Assigns cross-validation folds
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/22/2026
# ==============================================================================

import argparse as ap

from pathlib import Path
from datetime import datetime

from chip_stroma.training.cross_validation import (
    assign_folds,
    assign_chip_labels,
    merge_fold_assignments
)
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import (
    load_chip_labels,
    load_patch_manifest, 
    save_patch_manifest
)

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "02_cross_validation.yaml")

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "02_cross_validation.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )  

    # Load the patch manifest and CHIP labels
    manifest = load_patch_manifest(path = config.paths.metadata.patch_manifest)
    chip_labels = load_chip_labels(path = config.paths.metadata.chip_labels)

    # Merge the CHIP labels into the manifest
    manifest = assign_chip_labels(
        manifest       = manifest, 
        labels         = chip_labels,
        positive_label = config.cross_validation.positive_label,
        negative_label = config.cross_validation.negative_label,
    )

    # 2. Assign patient-level stratified k-fold indices to the patch manifest
    fold_assignments = assign_folds(
        manifest   = manifest,
        k          = config.cross_validation.k,
        seed       = config.cross_validation.seed,
        n_pos_bins = config.cross_validation.n_pos_bins
    )

    # Merge the assignments with the patch manifest and save it
    manifest = merge_fold_assignments(manifest, fold_assignments)

    save_patch_manifest(manifest, path = config.paths.metadata.patch_manifest)

    log_footer(cfg = config.paths)

    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Assign patches to CV folds.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Fold Assignment")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


def log_footer(cfg):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Patch Manifest: {cfg.metadata.patch_manifest}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]