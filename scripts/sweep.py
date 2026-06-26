# ==============================================================================
# Script:           sweep.yaml
# Purpose:          WandB sweep agent entrypoint
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import argparse as ap

from pathlib import Path
from datetime import datetime

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.training.train import train

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / f"{args.version}.yaml",
               version     = args.version)
    
    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "segmentation.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # 2. Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )
    



# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Begin a hyperparameter sweep.")
    parser.add_argument("--config_dir", type = str, default = "configs/sweeps/")
    parser.add_argument("--version", type = str, default = "v1")
    
    return parser.parse_args()


def log_header(config_path: Path, version: str):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Segmentation UNet Training - Sweep")
    logger.info(f"- Version: {version}")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

def log_footer(cfg, metrics):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Train Manifest: {cfg.metadata.train_manifest}")
    logger.info(f"- Validation Loss: {metrics.get('val/loss')}")
    logger.info(f"- Validation Dice Score: {metrics.get('val/dice')}")
    logger.info(f"- Validation IoU Score: {metrics.get('val/iou')}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]