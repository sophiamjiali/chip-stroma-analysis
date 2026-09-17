# ==============================================================================
# Script:           overlays.py
# Purpose:          Key stitched-together full WSI overlays
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/21/2026
# ==============================================================================

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from typing import Callable
from skimage.color import separate_stains
from skimage.filters import threshold_otsu
from skimage.color.colorconv import hdx_from_rgb

from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import SampleCoords, load_tissue_mask

logger = setup_logger(__name__)


# =====| General Helpers |======================================================

def place_patches(sample_id   : str,
                  coordinates : pd.DataFrame,
                  patch_size  : int,
                  slide_height: int,
                  slide_width : int,
                  get_patch   : Callable[[pd.Series], np.ndarray | None],
                  dtype       : np.typing.DTypeLike = np.float32
                 ) -> np.ndarray: 
    """
    Places non-overlapping patches on to the full WSI slide canvas.
    """

    # Initialize a canvas for stitching and a boolean tracker
    slide     = np.zeros((slide_height, slide_width), dtype = dtype)
    n_missing = 0

    # Place each row possessed in the coordinate table
    for _, row in coordinates.iterrows():
        patch = get_patch(row)
        if patch is None: 
            n_missing += 1
            continue

        y0, x0 = int(row['y']), int(row['x'])
        y1     = min(y0 + patch_size, slide_height)
        x1     = min(x0 + patch_size, slide_width)

        slide[y0:y1, x0:x1]   = patch[:y1 - y0, :x1 - x0]

    n = len(coordinates)
    logger.info(f"- Stitched {n - n_missing} / {n} patches")

    return slide


# =====| Stitch Vessel Predictions |============================================

def stitch_predictions(sample_id  : str,
                       predictions: dict[str, np.ndarray],
                       coordinates: SampleCoords,
                       threshold  : float) -> tuple[np.ndarray, np.ndarray]:
    """
    Stitches per-patch predicted vessel content to the WSI slide level. Returns 
    a map and binary mask.
    """

    def get_prediction_patch(row: pd.Series) -> np.ndarray | None:
        """Processes an individual patch. Accomodates if the patches extracted don't include coordinates, but are mapped in the metadata."""

        sanitized_name = re.sub(r"_x\d+_y\d+", "", row['patch_name'])
        print(sanitized_name)
        return predictions.get(sanitized_name)

    vessel_map = place_patches(
        sample_id    = sample_id,
        coordinates  = coordinates.table,
        patch_size   = coordinates.patch_size,
        slide_height = coordinates.slide_height,
        slide_width  = coordinates.slide_width,
        get_patch    = get_prediction_patch,
        dtype        = np.float32
    )
    vessel_mask = (vessel_map >= threshold).astype(np.uint8)

    return vessel_map, vessel_mask


# =====| Stitch Fibroblast Content |============================================

def stitch_fibroblast(sample_id  : str,
                      predictions: dict[str, np.ndarray],
                      coordinates: SampleCoords,
                      patch_dir  : Path,
                      mask_dir   : Path,
                      threshold  : float) -> np.ndarray:
    """
    Stitches per-patch fibroblast content to the WSI slide level. Returns 
    a map and binary mask.
    """

    def get_fibroblast_patch(row: pd.Series) -> np.ndarray | None:
        """Processes an individual patch."""

        # Load the patch tissue mask
        tissue = load_tissue_mask(
            sample_id  = sample_id,
            src_dir    = mask_dir,
            tile_id    = row['tile_id']
        )

        vessel_prob = predictions.get(row['patch_name'])
        image_path  = patch_dir / sample_id / row['patch_name']

        return quantify_fibroblast(image_path, vessel_prob, tissue, threshold)

    fibroblast_mask = place_patches(
        sample_id    = sample_id,
        coordinates  = coordinates.table,
        patch_size   = coordinates.patch_size,
        slide_height = coordinates.slide_height,
        slide_width  = coordinates.slide_width,
        get_patch    = get_fibroblast_patch,
        dtype        = np.uint8
    )

    return fibroblast_mask.astype(np.uint8)


def quantify_fibroblast(image_path : Path,
                        vessel_prob: np.ndarray | None,
                        tissue_mask: np.ndarray | None,
                        threshold  : float) -> np.ndarray | None: 
    """
    Extracts the fibroblast content of a single patch. Mirrors quantify_patch().
    """

    if tissue_mask is None or vessel_prob is None: return None

    vessel_mask = vessel_prob >= threshold
    valid_area = tissue_mask & ~vessel_mask

    # Reject degenerate case where all tissue content is vessel content
    if valid_area.sum() == 0: 
        return np.zeros_like(vessel_mask, dtype = np.uint8)

    # Extract fibroblast content via stain deconvolution of DAB stain
    with Image.open(image_path) as img: patch = np.array(img)
    dab_channel = separate_stains(patch, hdx_from_rgb)[:, :, 1]

    # Otsu restrict to valid pixels only
    otsu_t = threshold_otsu(dab_channel[valid_area])
    fibroblast_mask = (dab_channel >= otsu_t) & valid_area

    return fibroblast_mask.astype(np.uint8)


# =====| Stitch Tissue Mask |===================================================

def stitch_tissue_mask(sample_id  : str,  
                       coordinates: SampleCoords,
                       mask_dir   : Path) -> np.ndarray:
    """
    Stitches per-patch tissue mask to the WSI slide level. Returns 
    a map and binary mask.
    """

    def get_tissue_patch(row: pd.Series) -> np.ndarray | None:
        tissue = load_tissue_mask(
            sample_id  = sample_id,
            src_dir    = mask_dir,
            tile_id    = row['tile_id']
        )

        return None if tissue is None else tissue.astype(np.uint8)

    tissue_mask = place_patches(
        sample_id    = sample_id,
        coordinates  = coordinates.table,
        patch_size   = coordinates.patch_size,
        slide_height = coordinates.slide_height,
        slide_width  = coordinates.slide_width,
        get_patch    = get_tissue_patch,
        dtype        = np.uint8
    )

    return tissue_mask.astype(np.uint8)
      

# [END]