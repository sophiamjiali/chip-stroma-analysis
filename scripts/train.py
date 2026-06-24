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
import lightning.pytorch as pl

from pathlib import Path
from datetime import datetime

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.data.lit_module import VesselSegModule
from chip_stroma.data.datamodule import VesselSegmentationDataModule
from chip_stroma.segmentation.model import build_model

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "segmentation.yaml")

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

    # Seed everything using the random state provided from configurations
    seed = config.segmentation.data.seed
    logger.info(f"Seeded all processes and workers with random state {seed}")
    pl.seed_everything(seed, workers = True)

    # 3. Initialize the model and all components for training
    log_initialization_header(config)

    
    log_initialization_footer()

    # Initialize the output directly; one subdirectory per run...




    


    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Process raw patches and masks.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Segmentation UNet Training")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

def log_initialization_header(cfg):
    logger.info("=" * 50)
    logger.info("Step 03: Model Initialization")
    logger.info(f"- Project: {cfg.study.project}")
    logger.info(f"- Group: {cfg.study.group}")
    logger.info(f"- Encoder Name: {cfg.model.encoder_name}")
    logger.info(f"- Encoder Weight: {cfg.model.encoder_weights}")
    logger.info("-" * 50)

def log_initialization_footer():
    logger.info("=" * 50)

def log_footer(cfg):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Train Manifest: {cfg.metadata.train_manifest}")


    # ... include more




    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]