# ==============================================================================
# Script:           evaluate.py
# Purpose:          Model evaluation on the validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import torch

import pandas as pd
import argparse as ap

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from chip_stroma.models.model import VesselSegModule, build_model
from chip_stroma.data.dataset import VesselPatchDataset
from chip_stroma.data.transforms import get_val_transforms
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import load_patch_manifest
from chip_stroma.models.loss import dice_score

logger = setup_logger(__name__)


def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "evaluate.yaml")

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "evaluate.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # Load the model from the specified checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        encoder_name    = config.evaluate.model.encoder_name,
        encoder_weights = config.evaluate.model.encoder_weights,
        in_channels     = config.evaluate.model.in_channels,
        out_classes     = config.evaluate.model.out_classes
    )

    model = VesselSegModule.load_from_checkpoint(
        checkpoint_path = config.evaluate.checkpoint_path,
        map_location    = device,
        model           = model
    )
    model.to(device)
    model.eval()
    model.freeze()

    # Load the validation fold as a dataset
    manifest = load_patch_manifest(path = config.paths.metadata.patch_manifest)
    manifest = (manifest[manifest['fold'] == int(config.evaluate.val_fold)]
                .reset_index(drop = True))
    
    dataset = VesselPatchDataset(
        manifest        = manifest,
        patch_dir       = config.paths.processed_data.patch_dir,
        vessel_mask_dir = config.paths.processed_data.vessel_mask_dir,
        tissue_mask_dir = config.paths.processed_data.vessel_mask_dir,
        transform       = get_val_transforms()
    )
    dataloader = DataLoader(
        dataset     = dataset,
        batch_size  = config.evaluate.batch_size,
        num_workers = config.evaluate.n_workers,
        shuffle     = False,
        pin_memory  = True
    )

    # Compute per-patch metrics for the full validation cohort
    per_patch_results = []
    with torch.no_grad():
        for batch in dataloader:
            logger.info("Processing a batch...")
            
            # Fetch all information returned from the patch dataset
            images       = batch['patch'].to(device)
            vessel_masks = batch["vessel_mask"].squeeze(1).long().to(device)
            tissue_masks = batch["tissue_mask"].long().to(device)
            sample_ids   = batch["sample_id"]
            patch_names  = batch['patch_name']

            logits_sq = model(images).squeeze(1)
            preds = (torch.sigmoid(logits_sq) > 0.5).long()

            # Restrict to tissue regions only
            preds_m        = preds * tissue_masks
            vessel_masks_m = vessel_masks * tissue_masks

            # Exclude patches without positive vessel annotations
            has_signal = vessel_masks_m.sum(dim = (1, 2)) > 0

            # Compute per-sample Dice guarded against NaN
            dice_batch = dice_score(preds_m.float(), vessel_masks_m.float())

            # Compute precision/recall guarded against NaN
            tp = (preds_m * vessel_masks_m).flatten(1).sum(dim = 1)
            pred_pos = preds_m.flatten(1).sum(dim = 1)
            true_pos = vessel_masks_m.flatten(1).sum(dim = 1)
            
            precision = torch.where(pred_pos > 0, tp / pred_pos, 
                                    torch.tensor(float('nan')))
            recall    = torch.where(true_pos > 0, tp / true_pos, 
                                    torch.tensor(float('nan')))
            
            # Mask for patches with positive vessel annotations
            precision = torch.where(has_signal, precision, 
                                    torch.tensor(float('nan')))
            recall    = torch.where(has_signal, recall, 
                                    torch.tensor(float('nan')))
            
            # Move entire batch to CPU/numpy once
            dice_np       = dice_batch.detach().cpu().numpy()
            precision_np  = precision.detach().cpu().numpy()
            recall_np     = recall.detach().cpu().numpy()
            has_signal_np = has_signal.detach().cpu().numpy()

            for i in range(len(patch_names)):
                per_patch_results.append({
                    'sample_id' : sample_ids[i],
                    'patch_name': patch_names[i],
                    'dice'      : dice_np[i],
                    'precision' : precision_np[i],
                    'recall'    : recall_np[i],
                    'has_signal': has_signal_np[i],
                })

    # Aggregate patch-level results to sample-level; exclude no-signal patches
    per_patch_results = pd.DataFrame(per_patch_results)
    patch_signal = per_patch_results[per_patch_results['has_signal']]

    per_sample_metrics = (patch_signal.groupby('patient_id')
                          [['dice', 'precision', 'recall']].mean())
    
    logger.info(f"N patches (fold 0): {len(per_patch_results)} | "
                f"N positive-signal patches: {len(patch_signal)}")
    logger.info(f"N samples: {per_sample_metrics.shape[0]}")
    logger.info("\nPer-sample mean metrics:\n", per_sample_metrics)
    logger.info(f"\nMean Dice (sample-level): "
                f"{per_sample_metrics['dice'].mean():.4f}"
                f" ± {per_sample_metrics['dice'].std():.4f}")
    
    # Save all results
    dst_dir = config.paths.results.evaluation / Path(args.version)
    dst_dir.mkdir(parents = True, exist_ok = True)

    per_patch_results.to_csv(dst_dir / "patch_metrics.csv", index = False)
    per_sample_metrics.to_csv(dst_dir / "sample_metrics.csv", index = False)

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