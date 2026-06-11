# ==============================================================================
# Script:           preprocess.py
# Purpose:          Patch extraction and mask alignment
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import logging

import argparse as ap

from pathlib import Path
from datetime import datetime

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.io import (
    sanitize_names, 
    save_name_mapping,
    build_patch_manifest,
    save_patch_manifest
)
from chip_stroma.data.preprocessing import (
    apply_tissue_filter,
    apply_artifact_filter,
    fit_normalizer,
    normalize_patches
)

logger = logging.getLogger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(config_path = Path(args.config_dir) / "preprocess.yaml")

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "preprocess.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # 2. Sanitize sample folder names
    name_mapping = sanitize_names(src_dir = config.paths.raw_data.patch_dir)

    # 3. Initialize a patch manifest to log filtering and mapping
    manifest = build_patch_manifest(
        src_dir      = config.paths.raw_data.patch_dir,
        name_mapping = name_mapping
    )
    
    # 4. Detect tissue and remove background
    tissue_detection_cfg = config.preprocess.tissue_detection
    manifest = apply_tissue_filter(
        src_dir           = config.paths.raw_data.patch_dir,
        manifest          = manifest,
        tissue_threshold  = tissue_detection_cfg.tissue_threshold,
        gaussian_sigma    = tissue_detection_cfg.gaussian_sigma,
        min_region_size   = tissue_detection_cfg.min_region_size,
        morph_disk_radius = tissue_detection_cfg.morph_disk_radius,
        haem_min_signal   = tissue_detection_cfg.haem_min_signal,
        dab_min_signal    = tissue_detection_cfg.dab_min_signal
    )
    
    # 3. Detect artifacts and discard corrupted patches
    artifact_detection_cfg = config.preprocess.artifact_detection
    manifest = apply_artifact_filter(
        src_dir              = config.paths.raw_data.patch_dir,
        manifest             = manifest,
        blur_threshold       = artifact_detection_cfg.blur_threshold,
        dark_pixel_threshold = artifact_detection_cfg.dark_pixel_threhsold,
        dark_pixel_ratio     = artifact_detection_cfg.dark_pixel_ratio,
        pen_pixel_ratio      = artifact_detection_cfg.pen_pixel_ratio
    )

    # Save metadata generated during preprocessing before normalization
    save_name_mapping( name_mapping, path = config.paths.metadata.name_mapping)
    save_patch_manifest(manifest, path = config.paths.metadata.patch_manifest)

    # 4. Perform stain normalization
    included = manifest[manifest['include'] == True][['sample_id', 'patch']]
    normalizer = fit_normalizer(
        reference_path = config.preprocess.normalization.reference_patch,
        method         = config.preprocess.normalization.method
    )

    normalize_patches(
        included_patches = included,
        normalizer       = normalizer,
        src_patch_dir    = config.paths.raw_data.patch_dir,
        src_mask_dir     = config.paths.raw_Data.mask_dir,
        dst_patch_dir    = config.paths.processed_data.patch_dir,
        dst_mask_dir     = config.paths.processed_data.mask_dir
    )

    log_footer(cfg = config.paths)

    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Process raw patches and masks.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: Preprocessing")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

def log_footer(cfg):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Patch Directory: {cfg.processed_data.patch_dir}")
    logger.info(f"- Mask Directory: {cfg.processed_data.mask_dir}")
    logger.info(f"- Name Mapping: {cfg.metadata.name_mapping}")
    logger.info(f"- Patch Manifest: {cfg.metadata.patch_manifeset}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]