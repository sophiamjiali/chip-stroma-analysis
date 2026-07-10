# ==============================================================================
# Script:           visualize.yaml
# Purpose:          Generates overlays on representative patches
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import torch

import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from chip_stroma.models.model import VesselSegModule, build_model
from chip_stroma.data.dataset import VesselPatchDataset
from chip_stroma.data.transforms import get_val_transforms
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.models.loss import dice_score

logger = setup_logger(__name__)

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "visualize.yaml")

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "visualize.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # Load the model from the specified checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        encoder_name    = config.visualize.model.encoder_name,
        encoder_weights = config.visualize.model.encoder_weights,
        in_channels     = config.visualize.model.in_channels,
        out_classes     = config.visualize.model.out_classes
    )

    model = VesselSegModule.load_from_checkpoint(
        checkpoint_path = config.visualize.checkpoint_path,
        map_location    = device,
        model           = model
    )
    model.to(device)
    model.eval()
    model.freeze()

    # Load the validation fold as a dataset
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )
    manifest = (manifest[manifest['fold'] == int(config.visualize.val_fold)]
                .reset_index(drop = True))
    
    dataset = VesselPatchDataset(
        manifest        = manifest,
        patch_dir       = config.paths.processed_data.patch_dir,
        vessel_mask_dir = config.paths.processed_data.vessel_mask_dir,
        tissue_mask_dir = config.paths.processed_data.tissue_mask_dir,
        transform       = get_val_transforms()
    )

    # Load per-patch metrics to select top-k candidates
    metrics = pd.read_csv(config.paths.results.evaluation / 
                          Path(args.version) / Path("patch_metrics.csv"))

    # Stratified selection: best positives, worst false-positive tiles, etc.
    top_positive  = metrics[metrics["has_signal"]].nlargest(6, "dice")
    worst_fp      = metrics[metrics["has_signal"]].nsmallest(6, "precision")
    true_negative = metrics[~metrics["has_signal"]].sample(6, random_state = 0)

    selected = (pd.concat([top_positive, worst_fp, true_negative])
                .reset_index(drop = True))
    
    # Build overlays for all selected patches
    dst_dir = config.paths.results.overlays.vessels_dir
    dst_dir.mkdir(parents = True, exist_ok = True)

    with torch.no_grad():
        for _, row in selected.iterrows():

            # Select the index of the patch
            idx = (manifest.index[manifest['patch_name'] == 
                                  row.get('patch_name', None)])
            idx = (idx[0] if len(idx) else 
                   metrics.index[metrics['sample_id'] == row['sample_id']][0])
            
            # Extract patch data from the dataset
            sample = dataset[idx]
            image       = sample["patch"].unsqueeze(0).to(device)
            vessel_mask = sample["vessel_mask"].long().to(device)
            tissue_mask = sample["tissue_mask"].long().to(device)

            logits = model(image).squeeze()
            pred = (torch.sigmoid(logits) > 0.5).long() * tissue_mask
            target = vessel_mask * tissue_mask

            # Undo normalization upon the patch for display
            raw = image.squeeze(0).permute(1, 2, 0).numpy()
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

            # Generate an overlay of the prediction upon the ground-truth
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(raw); axes[0].set_title("Raw H-DAB"); axes[0].axis("off")
            axes[1].imshow(raw); axes[1].imshow(target.numpy(), cmap="Reds", alpha=0.4)
            axes[1].set_title("Ground truth"); axes[1].axis("off")
            axes[2].imshow(raw); axes[2].imshow(pred.numpy(), cmap="Blues", alpha=0.4)
            axes[2].set_title(f"Prediction (Dice={row['dice']:.2f})"); axes[2].axis("off")

            fig.suptitle(f"Sample {row['sample_id']}")
            fig.tight_layout()


            fig.savefig(dst_dir / f"{row['sample_id']}_{idx}.png", dpi=300)
            plt.close(fig)

            logger.info("- | - Saved an overlay")

    log_footer()
    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Train a single run of the model.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version", type = str)
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Model Evaluation")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

def log_footer():
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]