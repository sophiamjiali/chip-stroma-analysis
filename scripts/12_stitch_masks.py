# ==============================================================================
# Script:           12_stitch_masks.yaml
# Purpose:          Stitch together key overlays
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/21/2026
# ==============================================================================

import argparse as ap

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest

from chip_stroma.visualize.overlays import stitch_mask
from chip_stroma.utils.io import load_predictions

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():

    args = parse_args()
    log_header(
        pipeline_stage = "Analysis",
        config_path    = Path(args.config_dir) / "11_analysis.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "11_analysis.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Load the validation fold as a dataset
    manifest = initialize_train_manifest(
        train_path = config.paths.metadata.train_manifest,
        patch_path = config.paths.metadata.patch_manifest
    )

    # Extract key input and output directories
    patch_dir     = Path(config.paths.processed_data.patch_dir)
    coord_dir     = Path(config.paths.raw_data.patch_coords)
    inference_dir = Path(config.paths.results) / args.version / "inference"
    mask_dir      = Path(config.paths.results) / args.version / "stitch_masks"
    mask_dir.mkdir(parents = True, exist_ok = True)

    for _, row in manifest.iterrows():
        sample_id   = row['sample_id']
        original_id = row['original_id']
        patch_name  = row['patch_name']
        fold        = row['fold']

        # Load the patch and prediction mask from previous steps
        patch, mask = load_predictions(
            patch_dir = patch_dir,
            pred_dir  = inference_dir,
            fold      = fold,
            sample_id = sample_id
        )




# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Mask overlay stitching.")
    parser.add_argument("--config_dir",   type = str, default = "configs/")
    parser.add_argument("--version",      type = str)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]