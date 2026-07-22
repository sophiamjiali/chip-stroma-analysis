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

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "08_visualize.yaml")

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "08_visualize.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Load the model from the specified checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        encoder_name    = config.visualize.model.encoder_name,
        encoder_weights = config.visualize.model.encoder_weights,
        in_channels     = config.visualize.model.in_channels,
        out_classes     = config.visualize.model.out_classes
    )

    ckpt_path = config.paths.checkpoints / args.version / "best_trial.ckpt"
    model = VesselSegModule.load_from_checkpoint(
        checkpoint_path = ckpt_path,
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
    dst_dir = config.paths.evaluation / Path(args.version)
    metrics = pd.read_csv(dst_dir / Path("patch_metrics.csv"))

    # Stratified selection: best positives, worst false-positive tiles, etc.
    positive = metrics[metrics['has_signal']]
    negative = metrics[~metrics['has_signal']]

    true_positive      = positive.nlargest(args.n_plots, "dice")
    false_negative     = positive.nsmallest(args.n_plots, "recall")
    false_positive     = negative.nlargest(args.n_plots, "dice")
    true_negative      = negative.nsmallest(args.n_plots, "dice")
    over_segmentation  = positive.nsmallest(args.n_plots, "precision")
    under_segmentation = positive.nsmallest(args.n_plots, "recall")

    true_positive      = true_positive.assign(category = "true_positive")
    false_negative     = false_negative.assign(category = "false_negative")
    false_positive     = false_positive.assign(category = "false_positive")
    true_negative      = true_negative.assign(category = "true_negative")
    over_segmentation  = over_segmentation.assign(category = "over_seg")
    under_segmentation = under_segmentation.assign(category = "under_seg")

    selected = (
        pd.concat([true_positive, false_negative, false_positive, 
                   true_negative, over_segmentation, under_segmentation])
                   .reset_index(drop = True)
    )
    
    # Build overlays for all selected patches
    for category in ['true_positive', 'false_negative', 'false_positive', 
                     'true_negative', 'over_seg', 'under_seg']:
        (dst_dir / category).mkdir(parents = True, exist_ok = True)

    with torch.no_grad():
        for _, row in selected.iterrows():

            # Select the index of the patch
            match = manifest.index[
                (manifest['sample_id'] == row['sample_id']) &
                (manifest['patch_name'] == row['patch_name'])
            ]
            idx = match[0]
            
            # Extract patch data from the dataset
            sample = dataset[idx]
            image       = sample['patch'].unsqueeze(0).to(device)
            vessel_mask = sample["vessel_mask"].long().to(device)
            tissue_mask = sample["tissue_mask"].long().to(device)

            logits = model(image).squeeze()
            pred = (torch.sigmoid(logits) > 0.5).long() * tissue_mask
            target = vessel_mask * tissue_mask

            # Undo normalization upon the patch for display
            raw = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

            # Generate an overlay of the prediction upon the ground-truth
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(raw); axes[0].set_title("Raw H-DAB"); axes[0].axis("off")
            axes[1].imshow(raw); axes[1].imshow(target.squeeze(0).cpu().numpy(), cmap="Reds", alpha=0.6)
            axes[1].set_title("Ground truth"); axes[1].axis("off")
            axes[2].imshow(raw); axes[2].imshow(pred.squeeze(0).cpu().numpy(), cmap="Blues", alpha=0.6)
            axes[2].set_title(f"Prediction (Dice={row['dice']:.2f})"); axes[2].axis("off")

            # Label the panel with its stratification category
            fig.suptitle(f"Sample {row['sample_id']} — {row['category'].replace('_', ' ').title()}")
            fig.tight_layout()

            # Save into the corresponding category subfolder
            fig.savefig(dst_dir / row['category'] / f"{row['sample_id']}_{idx}.png", dpi=300)
            plt.close(fig)
            logger.info(f"- | - Saved a {row['category']} overlay")

    log_footer()
    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Train a single run of the model.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version", type = str)
    parser.add_argument("--n_plots", type = int, default = 6)
    
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