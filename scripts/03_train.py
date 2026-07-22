# ==============================================================================
# Script:           train.yaml
# Purpose:          Train the UNet for vessel train
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import argparse as ap

from pathlib import Path

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.training.train import train

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Training",
        config_path    = Path(args.config_dir) / "03_train.yaml"
    )

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "03_train.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # 2. Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )

    # Train the model using the configurations provided in the YAML
    metrics = train(
        manifest = manifest,
        project  = config.train.study.project,
        group    = config.train.study.group,
        paths    = config.paths,
        params   = config.train,
        seed     = config.train.data.seed,
        trial    = None
    )

    logger.info("=" * 50)
    logger.info("Final Training Metrics:")
    logger.info(f"- Validation Loss: {metrics.get('val/loss')}")
    logger.info(f"- Validation Dice Score: {metrics.get('val/dice')}")
    logger.info(f"- Validation IoU Score: {metrics.get('val/iou')}")
    logger.info("=" * 50)

    log_footer()

    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Train a single run of the model.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]