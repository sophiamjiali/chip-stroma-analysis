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
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

JOIN_KEY = ['sample_id', 'patch_name']

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


# =====| CHIP Labels |==========================================================

def load_chip_labels(path: Path) -> pd.DataFrame:
    """Loads the patch manifest from the path."""
    return pd.read_csv(path)

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
                'sample_id':          sanitized_id,
                'original_id':        original_id,
                'patch_name':         patch_path.name,
                'vessel_mask':        None,
                'tissue_mask':        None,
                'chip_status':        None,
                'has_vessel':         None,
                'vessel_pixel_count': None,
                'passes_tissue':      None,
                'passes_artifact':    None,
                'include':            None,
                'norm_status':        None,
                'patient_pos_count':  None,
                'fold':               None,
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


def load_patch_manifest(path: Path) -> pd.DataFrame:
    """Loads the patch manifest from the path."""
    return pd.read_csv(path)

# =====| Patch Statistics |=====================================================

def build_patch_stats(src_dir: Path,
                      name_mapping: dict) -> pd.DataFrame:
    """
    Initialize a patch statistics manifest with one row per patch.
    All filter columns default to None — populated by downstream filter functions.
    """

    logger.info("=" * 50)
    logger.info("Step 04: Patch Statistics")
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
                'sample_id':          sanitized_id,
                'patch_name':         patch_path.name,
                'tissue_ratio':       None,
                'lap_variance':       None,
                'dark_ratio':         None,
                'pen_ratio':          None
            })

    logger.info(f"Initialized statistics for {len(name_mapping)} samples")
    logger.info(f"Populated statistics with {total_patches} total patches")
    logger.info("=" * 50)

    return pd.DataFrame(records)


def save_patch_stats(stats: pd.DataFrame, path: Path) -> None:
    """Persist patch manifest to CSV."""

    path.parent.mkdir(parents = True, exist_ok = True)
    stats.to_csv(path, index = False)
    return None

# =====| Report Updates |=======================================================

def update_vessel_report(manifest: pd.DataFrame, 
                         vessel_report: pd.DataFrame) -> pd.DataFrame:
    """Updates the manifest with the vessel report."""

    vessel_lookup = vessel_report.set_index(JOIN_KEY)

    manifest['vessel_mask'] = (manifest.set_index(JOIN_KEY).index.map(
        vessel_lookup['vessel_mask'].to_dict()))
    
    manifest['has_vessel'] = (manifest.set_index(JOIN_KEY).index.map(
        vessel_lookup['has_vessel'].to_dict()))

    manifest['vessel_pixel_count'] = (manifest.set_index(JOIN_KEY).index.map(
        vessel_lookup['vessel_pixel_count'].to_dict()))

    return manifest


def update_tissue_report(manifest: pd.DataFrame,
                         statistics: pd.DataFrame, 
                         tissue_report: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Updates the manifest and statistics with the tissue report."""

    tissue_lookup = tissue_report.set_index(JOIN_KEY)

    manifest['passes_tissue'] = (manifest.set_index(JOIN_KEY).index.map(
        tissue_lookup['passes_tissue'].to_dict()))

    manifest['tissue_mask'] = (manifest.set_index(JOIN_KEY).index.map(
        tissue_lookup['tissue_mask'].to_dict()))

    manifest['include'] = manifest['passes_tissue']

    statistics['tissue_ratio'] = (statistics.set_index(JOIN_KEY).index.map(
        tissue_lookup['tissue_ratio'].to_dict()))

    return manifest, statistics


def update_artifact_report(manifest: pd.DataFrame,
                           statistics: pd.DataFrame, 
                           artifact_report: pd.DataFrame
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Updates the manifest statistics with the artifact report."""

    artifact_lookup = artifact_report.set_index(JOIN_KEY)

    manifest['passes_artifact'] = (manifest.set_index(JOIN_KEY).index.map(
        artifact_lookup['passes_artifact'].to_dict()))

    mask = (
        (manifest['include']) &
        (~manifest['passes_artifact'])
    )
    manifest.loc[mask, 'include'] = False

    statistics['lap_variance'] = (statistics.set_index(JOIN_KEY).index.map(
        artifact_lookup['lap_variance'].to_dict()))
    
    statistics['dark_ratio'] = (statistics.set_index(JOIN_KEY).index.map(
        artifact_lookup['dark_ratio'].to_dict()))
    
    statistics['pen_ratio'] = (statistics.set_index(JOIN_KEY).index.map(
        artifact_lookup['pen_ratio'].to_dict()))

    return manifest, statistics

# =====| Directory Cleanup |====================================================

def prune_tissue_masks(manifest: pd.DataFrame, 
                       src_mask_dir: Path,
                       n_workers: int | None = None) -> None:
    """Delete tissue masks for excluded patches."""

    excluded = manifest.loc[~manifest["include"], ["sample_id", "patch"]]

    mask_paths = [
        src_mask_dir / str(row.sample_id) / 
        str(row.patch).replace("_raw.png", "_tissue_mask.png") 
        for row in excluded.itertuples(index = False)
    ]

    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        list(tqdm(pool.map(delete_mask, mask_paths, chunksize = 100), 
                  total = len(mask_paths)))


def delete_mask(mask_path: Path) -> None:
    if mask_path.exists(): mask_path.unlink()

# [END]