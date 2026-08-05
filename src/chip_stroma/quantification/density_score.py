# ==============================================================================
# Script:           density_score.py
# Purpose:          Compute aSMA+ and vessel region area per patient
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/04/2026
# ==============================================================================

import h5py

import numpy as np
import pandas as pd

from box import Box
from typing import cast
from pathlib import Path
from skimage.color import separate_stains, hdx_from_rgb
from skimage.filters import threshold_otsu

from chip_stroma.data.transforms import get_val_transforms
from chip_stroma.data.dataset import VesselPatchDataset
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import (
    build_fold_manifest, 
    save_overlay_masks, 
    save_overlay_patch
)

logger = setup_logger(__name__)
CHUNK_SIZE = 512


def quantify_fold(fold        : int,
                  manifest    : pd.DataFrame,
                  thresholds  : list[float],
                  base_thresh : float,
                  mask_dir    : Path,
                  paths       : Box,
                  version     : str,
                  single_model : bool = False) -> list[dict]:
    """
    Run quantification for one fold (or the single all-data model), 
    index-matched to that fold's val_arrays.h5, following the exact split logic.
    """

    # Fetch the inference path from the inference step
    results_dir = Path(paths.results) / version / "inference"
    fold_dir    = results_dir if single_model else results_dir / f"fold_{fold}"

    # Initialize the fold's held-out validation split
    fold_manifest = build_fold_manifest(manifest, fold, single_model)
    dataset = VesselPatchDataset(
        manifest        = fold_manifest,
        patch_dir       = paths.processed_data.patch_dir,
        vessel_mask_dir = paths.processed_data.vessel_mask_dir,
        tissue_mask_dir = paths.processed_data.tissue_mask_dir,
        transform       = get_val_transforms()
    )

    rows = []
    with h5py.File(fold_dir / "val_arrays.h5", "r") as h5f:
        probs = cast(h5py.Dataset, h5f['probs'])
        n     = probs.shape[0]

        logger.info(f"- Loaded {n} fold predictions")

        # Stream in chunks, dequantizing uint8 -> [0,1] float32
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            chunk_probs = probs[start:end].astype(np.float32) / 255.0

            for i in range(start, end):
                item = dataset[i]
                prob = chunk_probs[i - start].squeeze()

                patch       = item['patch'].cpu().numpy()
                tissue_mask = item['tissue_mask'].cpu().numpy().astype(bool)

                patch_results = quantify_patch(
                    patch       = patch,
                    prob        = prob,
                    tissue_mask = tissue_mask,
                    thresholds  = thresholds
                )

                # Export the fibroblast mask for the provided threshold
                out_dir = mask_dir / item['sample_id']
                out_dir.mkdir(parents = True, exist_ok = True)

                patch_name     = item['patch_name'].removesuffix('_raw.png')
                out_mask_path  = out_dir / f"{patch_name}_masks.png"
                out_patch_path = out_dir / f"{patch_name}_overlay.png"

                qa_masks = patch_results[base_thresh]

                # Save masks as a palette
                save_overlay_masks(
                    vessel_mask     = qa_masks['vessel_mask'],
                    fibroblast_mask = qa_masks['fibroblast_mask'],
                    tissue_mask     = tissue_mask,
                    out_path        = out_mask_path
                )

                # Save masks overlaid on the patch itself
                save_overlay_patch(
                    patch           = patch,
                    vessel_mask     = qa_masks['vessel_mask'],
                    fibroblast_mask = qa_masks['fibroblast_mask'],
                    tissue_mask     = tissue_mask,
                    out_path        = out_patch_path
                )

                # Format quantification statistics
                for t, res in patch_results.items():
                    rows.append({
                        "sample_id"         : item["sample_id"],
                        "patch_name"        : item["patch_name"],
                        "fold"              : fold,
                        "vessel_threshold"  : t,
                        "fibroblast_density": res["density"]
                    })

    logger.info("- Quantified all patches")
    return rows


def quantify_patch(patch      : np.ndarray,
                   prob       : np.ndarray,
                   tissue_mask: np.ndarray,
                   thresholds : list[float]) -> dict[float, dict]: 
    """
    Compute fibroblast density at each vessel-threshold in the sensitivity 
    sweep.
    """

    # DAB deconvolution per patch
    patch       = np.moveaxis(patch, 0, -1)
    dab_channel = separate_stains(patch, hdx_from_rgb)[:, :, 1]

    results = {}
    for t in thresholds:
        vessel_mask = prob >= t
        valid_area = tissue_mask & ~vessel_mask

        # Detect degenerate cases; entire tissue masked as vessel
        if valid_area.sum() == 0:
            results[t] = {
                'density'        : np.nan,
                'vessel_mask'    : vessel_mask,
                'fibroblast_mask': np.zeros_like(vessel_mask)
            }
            continue

        # Otsu restricted to valid pixels only
        otsu_t = threshold_otsu(dab_channel[valid_area])
        fibroblast_mask = (dab_channel >= otsu_t) & valid_area

        density = fibroblast_mask.sum() / valid_area.sum()
        results[t] = {
            'density'        : density,
            'vessel_mask'    : vessel_mask,
            'fibroblast_mask': fibroblast_mask
        }

    return results


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Macro-average patch densities to sample-level, per vessel threshold."""

    return (
        scores.groupby(['sample_id', 'vessel_threshold'])['fibroblast_density']
        .mean()
        .reset_index()
        .rename(columns = {'fibroblast_density': 'sample_fibroblast_density'})
    )


def summarize_sensitivity(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-patient density range/CV across the threshold sweep."""

    summary = scores.groupby('sample_id')['sample_fibroblast_density'].agg(
        density_min = "min", density_max = "max", 
        density_mean = "mean", density_std = "std"
    )
    summary['density_range'] = summary['density_max'] - summary['density_min']
    summary['density_cv'] = summary['density_std'] / summary['density_mean']

    return summary.reset_index()

# [END]