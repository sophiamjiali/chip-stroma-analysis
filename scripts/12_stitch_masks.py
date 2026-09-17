# ==============================================================================
# Script:           12_stitch_masks.yaml
# Purpose:          Stitch together key overlays
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/21/2026
# ==============================================================================

import json

import argparse as ap
import numpy as np

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest

from chip_stroma.visualize.overlays import (
    stitch_predictions,
    stitch_fibroblast,
    stitch_tissue_mask
)

from chip_stroma.utils.io import (
    load_coordinates, 
    load_predictions,
    save_vessel_heatmap,
    save_mask_png,
    mask_to_geojson
)

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():

    args = parse_args()
    log_header(
        pipeline_stage = "Stitch Masks",
        config_path    = Path(args.config_dir) / "12_stitch_masks.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "12_stitch_masks.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Load the validation fold as a dataset
    manifest = initialize_train_manifest(
        train_path = config.paths.metadata.train_manifest,
        patch_path = config.paths.metadata.patch_manifest
    )

    # Extract key input and output directories
    patch_dir     = Path(config.paths.processed_data.patch_dir)
    coord_dir     = Path(config.paths.raw_data.patch_coords_dir)
    tissue_dir    = Path(config.paths.processed_data.tissue_mask_dir)
    inference_dir = Path(config.paths.results) / args.version / "inference"
    mask_dir      = Path(config.paths.results) / args.version / "stitch_masks"
    mask_dir.mkdir(parents = True, exist_ok = True)

    colours = config.stitch_masks.colours

    # Load the mapping for sanitized to unsanitized sample IDs
    # name_mapping = load_json(config.paths.metadata.name_mapping)

    for _, row in manifest.iterrows():
        sample_id  = row['sample_id']
        patch_name = row['patch_name']
        fold       = row['fold']

        # # Fetch the unsanitized sample ID to map back to the coordinates
        # raw_sample_id = next(key for key, value in name_mapping.items() 
        #                      if value == sample_id)

        logger.info(f"Beginning to process item: {sample_id} - {patch_name}")

        # Load the sample's patch coordinates
        patch_coords = load_coordinates(
            sample_id    = sample_id,
            coord_dir    = coord_dir
        )

        logger.info(f"- Identified {len(patch_coords.table)} coordinates")

        # Load the model's vessel predictions
        vessel_probs = load_predictions(
            sample_id = sample_id,
            fold      = fold,
            pred_dir  = inference_dir
        )
        logger.info(f"- Loaded vessel predictions")

        # Stitch the vessel prediction mask into the full WSI
        vessel_map, vessel_mask = stitch_predictions(
            sample_id   = sample_id,
            predictions = vessel_probs,
            coordinates = patch_coords,
            threshold   = config.stitch_masks.vessel_threshold
        )
        logger.info("- Stitched vessel prediction mask")

        # Derive and stitch fibroblast mask into the full WSI
        fibro_mask = stitch_fibroblast(
            sample_id   = sample_id,
            predictions = vessel_probs,
            coordinates = patch_coords,
            patch_dir   = patch_dir,
            mask_dir    = mask_dir,
            threshold   = config.stitch_masks.vessel_threshold
        )
        logger.info("- Stitched fibroblast prediction mask")

        # Stitch the tissue mask into the full WSI
        tissue_mask = stitch_tissue_mask(
            sample_id   = sample_id,
            coordinates = patch_coords,
            mask_dir    = tissue_dir
        )
        logger.info("- Stitched tissue mask")

        # Save probability and mask NumPy files for downstream analysis
        out_dir = mask_dir / sample_id
        out_dir.mkdir(parents = True, exist_ok = True)

        np.save(out_dir / "vessel_prob.npy", vessel_map.astype(np.float16))
        np.save(out_dir / "vessel_mask.npy", vessel_mask)
        np.save(out_dir / "fibroblast_mask.npy", fibro_mask)
        np.save(out_dir / "tissue_mask.npy", tissue_mask)

        heatmap_path = out_dir / "vessel_heatmap.png"
        save_vessel_heatmap(vessel_map = vessel_map, path = heatmap_path)

        save_mask_png(vessel_mask, out_dir / "vessel_mask.png")
        save_mask_png(fibro_mask, out_dir / "fibroblast_mask.png")
        save_mask_png(tissue_mask, out_dir / "tissue_mask.png")

        vessel_gj = mask_to_geojson(vessel_mask, "vessel", colours.vessel)
        fibro_gj  = mask_to_geojson(fibro_mask, "fibroblast",colours.fibroblast)
        tissue_gj = mask_to_geojson(tissue_mask, "tissue", colours.tissue)

        (out_dir / "vessel.geojson").write_text(json.dumps(vessel_gj))
        (out_dir / "fibroblast.geojson").write_text(json.dumps(fibro_gj))
        (out_dir / "tissue.geojson").write_text(json.dumps(tissue_gj))

        logger.info("- Saved all key outputs")

    logger.info("Completed all stitching.")
    log_footer()
    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Mask overlay stitching.")
    parser.add_argument("--config_dir",   type = str, default = "configs/")
    parser.add_argument("--version",      type = str)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]