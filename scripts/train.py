# ==============================================================================
# Script:           train.yaml
# Purpose:          Train the UNet for vessel segmentation
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
#
# Notes:            Output directory layout (all under paths.output.training):
#
#                   outputs/training/
#                   └── fold_0/                        # one directory per fold
#                       ├── config.yaml                # merged config snapshot
#                       │   ├── checkpoints/
#                       │   ├── best-epoch_*.ckpt      # best checkpoint
#                       │   └── last.ckpt              # latest epoch
#                       └── wandb/                     # W&B offline run dir.
#                             └── run-<id>/             # auto-created by wandb
#                                    
#                   Sync W&B after HPC Job completes:
#                       wandb sync --sync-all
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
    log_header(config_path = Path(args.config_dir) / "train.yaml")

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "train.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # 2. Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )

    # Train the model using the configurations provided in the YAML
    metrics = train(
        manifest = manifest,
        project  = config.segmentation.study.project,
        group    = config.segmentation.study.group,
        paths    = config.paths,
        params   = config.segmentation,
        seed     = config.segmentation.data.seed,
        trial    = None
    )

    log_footer(cfg = config.paths, metrics = metrics)

    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Train a single run of the model.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Segmentation UNet Training - Single Run")
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