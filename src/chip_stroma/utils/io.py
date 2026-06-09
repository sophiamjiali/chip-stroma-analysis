# ==============================================================================
# Script:           paths.yaml
# Purpose:          Input and output utilities with sanitization
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import logging
import json

import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)

# =====| Name Sanitization |====================================================

def sanitize_names(src_dir: Path) -> dict:
    """Map original sample_id folder names to space-sanitized versions."""

    logger.info("=" * 50)
    logger.info("Step 02: Name Sanitization")
    logger.info(f"- Source Directory: {src_dir}")
    logger.info("-" * 50)

    mapping = {
        p.name: p.name.replace(" ", "_")
        for p in src_dir.iterdir()
        if p.is_dir()
    }
     
    logger.info(f"Sanitized {len(mapping)} sample IDs")
    logger.info("=" * 50)

    return mapping
    

def save_name_mapping(name_mapping: dict, path: Path) -> None:
    """Persist sanitized name mapping to JSON."""

    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w") as f:
        json.dump(name_mapping, f, indent = 2)
    return None


# =====| Patch Manifest |=======================================================

def build_patch_manifest(src_dir: Path, 
                         name_mapping: dict) -> pd.DataFrame:
    """
    Initialize a patch manifest with one row per patch.
    All filter columns default to None — populated by downstream filter functions.
    """

    logger.info("=" * 50)
    logger.info("Step 03: Patch Manifest")
    logger.info(f"- Source Directory: {src_dir}")
    logger.info("-" * 50)
    
    total_patches = 0
    records = []
    for original_id, sanitized_id in name_mapping.items():

        # Extract the raw paths to all patches associated with the sample
        patch_paths = sorted((src_dir / original_id).glob("*_raw.png"))
        total_patches += len(patch_paths)

        for patch_path in patch_paths:
            records.append({
                'sample_id': sanitized_id,
                'original_id': original_id,
                'patch': patch_path.name,
                'passes_tissue': None,
                'passes_artifacts': None,
                'include': None
            })

    logger.info(f"Initialized manifest for {len(name_mapping)} samples")
    logger.info(f"Populated manifest with {total_patches} total patches")
    logger.info("=" * 50)

    return pd.DataFrame(records)


def save_patch_manifest(manifest: pd.DataFrame, path: Path) -> None:
    """Persist patch manifest to CSV."""

    path.parent.mkdir(parents = True, exist_ok = True)
    manifest.to_csv(path, index = False)
    return None

# [END]