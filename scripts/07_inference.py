# ==============================================================================
# Script:           07_inference.py
# Purpose:          Perform model inference on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import torch
import pickle

import argparse as ap

from pathlib import Path

from torch.utils.data import DataLoader

from chip_stroma.models.model import VesselSegModule, build_model
from chip_stroma.data.dataset import VesselPatchDataset
from chip_stroma.data.transforms import get_val_transforms
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest

from chip_stroma.models.inference import infer_fold

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Evaluation",
        config_path    = Path(args.config_dir) / "07_evaluate.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "07_evaluate.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Initialize version results directory and checkpoints (full CV models)
    dst_dir  = Path(config.paths.results) / args.version / "inference"
    ckpt_dir = Path(config.paths.checkpoints.full_cv) / args.version

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the validation fold as a dataset
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )
    
    # Loop over all folds for full-CV models (or specify one fold)
    folds = ([config.inference.n_folds] if args.val_fold_only 
             else range(config.inference.n_folds))
    
    for fold in folds:
        logger.info(f"Running inference on fold {fold}")

        # Load the fold's final CV model 
        ckpt_path = ckpt_dir / f"fold_{fold}.ckpt"
        model = build_model(
            encoder_name    = config.inference.model.encoder_name,
            encoder_weights = config.inference.model.encoder_weights,
            in_channels     = config.inference.model.in_channels,
            out_classes     = config.inference.model.out_classes
        )

        model = VesselSegModule.load_from_checkpoint(
            checkpoint_path = ckpt_path,
            map_location    = device,
            model           = model
        )
        model.to(device).eval().freeze()
        
        # Initialize the fold's held-out validation split
        fold_manifest =manifest[manifest['fold'] == fold].reset_index(drop=True)
    
        dataset = VesselPatchDataset(
            manifest        = fold_manifest,
            patch_dir       = config.paths.processed_data.patch_dir,
            vessel_mask_dir = config.paths.processed_data.vessel_mask_dir,
            tissue_mask_dir = config.paths.processed_data.tissue_mask_dir,
            transform       = get_val_transforms()
        )
        dataloader = DataLoader(
            dataset     = dataset,
            batch_size  = int(config.inference.model.batch_size),
            num_workers = int(config.inference.model.n_workers),
            shuffle     = False,
            pin_memory  = True
        )

        # Infer using the model and tag the fold explicitly
        patch_metrics, probs, gt = infer_fold(model, dataloader, device)
        patch_metrics['fold'] = fold
        
        # Save raw artifacts without aggregation (08_evaluate)
        fold_dir = dst_dir / f"fold_{fold}"
        fold_dir.mkdir(parents = True, exist_ok = True)

        patch_metrics.to_csv(fold_dir / "patch_metrics.csv", index = False)
        with open(fold_dir / "val_probs.pkl", "wb") as f: pickle.dump(probs, f)
        with open(fold_dir / "val_gt.pkl", "wb") as f   : pickle.dump(gt,    f)

        logger.info(f"Fold {fold}: {len(patch_metrics)} patches processed")

    logger.info("Inference complete for all fold(s).")
    log_footer()

    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Model evaluation.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version", type = str)
    parser.add_argument("--val_fold_only", action = "store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()

# [END]