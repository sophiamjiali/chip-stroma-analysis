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
    normalize_patches
)

logger = logging.getLogger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()

    config = load_configs(
        pipeline = Path(args.config_dir) / "preprocess.yaml",
        paths    = Path(args.config_dir) / "paths.yaml"
    )

    # 1. Sanitize sample folder names
    name_mapping = sanitize_names(
        src_dir = config.paths.raw_data.raw_dir
    )

    # Initialize a patch manifest to log filtering and mapping
    manifest = build_patch_manifest(
        src_dir      = config.paths.raw_data.patch_dir,
        name_mapping = name_mapping
    )
    
    # 2. Detect tissue and remove background
    manifest = apply_tissue_filter(
        src_dir = config.paths.raw_data.patch_dir,
        manifest = manifest,
        tissue_threshold = config.preprocess.tissue_detection.tissue_threshold,
        gaussian_sigma = config.preprocess.tissue_detection.gaussian_sigma,
        min_region_size = config.preprocess.tissue_detection.min_region_size,
        morph_disk_radius = config.preprocess.tissue_detection.morph_disk_radius
    )
    
    # 3. Detect artifacts and discard corrupted patches
    patch_manifest = apply_artifact_filter(
        src_dir = config.paths.raw_data.patch_dir,
        manifest = manifest,
        ...
    )

    # Save metadata generated during preprocessing before normalization
    save_name_mapping( name_mapping, path = config.paths.metadata.name_mapping)
    save_patch_manifest(manifest, path = config.paths.metadata.patch_manifest)

    # 4. Perform stain normalization
    included = manifest[manifest['include'] == True][['sample_id', 'patch']]
    normalizer = fit_normalizer(
        reference_path = config.preprocess.normalization.reference_patch,
        method = config.preprocess.normalization.method
    )

    normalize_patches(
        included_patches = included,
        normalizer       = normalizer,
        src_patch_dir    = config.paths.raw_data.patch_dir,
        src_mask_dir     = config.paths.raw_Data.mask_dir,
        dst_patch_dir    = config.paths.processed_data.patch_dir,
        dst_mask_dir     = config.paths.processed_data.mask_dir,
        name_mapping     = name_mapping
    )


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Process raw patches and masks.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]