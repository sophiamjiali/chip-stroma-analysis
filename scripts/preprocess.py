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
    run_step, 
    mark_complete,
    step_complete,
    sanitize_names, 
    save_name_mapping,
    move_directory_contents
)
from chip_stroma.data.preprocessing import (
    detect_tissue,
    detect_artifacts,
    normalize_patch
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
        src_dir = config.paths.raw_data.raw_dir,
        out_dir = config.paths.intermediate_data.sanitized_dir
    )
    save_name_mapping(name_mapping, path = config.paths.metadata.name_mapping)
    mark_complete(config.paths.intermediate_data.sanitized_dir)


    # 2. Detect tissue and remove background
    run_step(
        src_dir = config.paths.intermediate_data.sanitized_dir,
        dst_dir = config.paths.intermediate_data.tissue_dir,
        process_fn = detect_tissue,
        cleanup_src = True,
        ...
    )

    # 3. Detect artifacts and discard corrupted patches
    run_step(
        src_dir = config.paths.intermediate_data.tissue_dir,
        dst_dir = config.paths.intermediate_data.artifact_dir,
        process_fn = detect_artifacts,
        cleanup_src = True,
        ...
    )

    # 4. Perform stain normalization
    run_step(
        src_dir = config.paths.intermediate_data.artifact_dir,
        dst_dir = config.paths.intermediate_data.normalized_dir,
        process_fn = normalize_patch,
        cleanup_src = True,
        ...
    )

    assert step_complete(config.paths.intermediate_data.normalized_dir), "Normalization was not completed successfully -- aborting cleanup"

    # Move all processed data into the processed directories
    move_directory_contents(
        src_dir = config.paths.intermediate_data.normalized_dir,
        dst_dir = config.paths.processed_data.processed_dir
    )


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Process raw patches and masks.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]