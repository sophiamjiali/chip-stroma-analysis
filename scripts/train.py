# ==============================================================================
# Script:           train.yaml
# Purpose:          Train the UNet for vessel segmentation
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import logging

import argparse as ap
import pytorch_lightning as pl

from pathlib import Path
from datetime import datetime

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.segmentation.lit_module import VesselSegModule
from chip_stroma.data.datamodule import VesselSegmentationDataModule
from chip_stroma.segmentation.model import build_model

logger = logging.getLogger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "segmentation.yaml")

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "segmentation.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )

    # Seed everything using the random state provided from configurations
    pl.seed_everything(config.segmentation.data.seed, workers = True)

    # Initialize the data module, model, and LightningModule
    datamodule = VesselSegmentationDataModule(
        manifest        = manifest,
        patch_dir       = config.paths.processed_data.patch_dir,
        vessel_mask_dir = config.paths.processed_data.vessel_mask_dir,
        tissue_mask_dir = config.paths.processed_data.tissue_mask_dir,
        fold            = config.segmentation.data.fold,
        batch_size      = config.segmentation.data.batch_size,
        num_workers     = config.segmentation.data.num_workers,
        pos_prob        = config.segmentation.data.pos_prob,
        seed            = config.segmentation.data.seed
    )

    model = build_model(
        encoder_name    = config.segmentation.model.encoder_name,
        encoder_weights = config.segmentation.model.encoder_weights,
        in_channels     = config.segmentation.model.in_channels,
        out_classes     = config.segmentation.model.out_channels
    )

    lit_module = VesselSegModule(
        model = model,
        lr           = config.segmentation.module.lr,
        weight_decay = config.segmentation.module.weight_decay,
        T_max        = config.segmentation.trainer.max_epochs,
        ftl_alpha    = config.segmentation.module.ftl_alpha,
        ftl_beta     = config.segmentation.module.ftl_beta,
        ftl_gamma    = config.segmentation.module.ftl_gamma,
        ftl_weight   = config.segmentation.module.ftl_weight,
        smooth       = config.segmentation.module.smooth
    )

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