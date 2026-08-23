# ==============================================================================
# Script:           overlays.py
# Purpose:          Key stitched-together full WSI overlays
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/21/2026
# ==============================================================================

from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from PIL import Image
from skimage.color import separate_stains
from skimage.color.colorconv import hdx_from_rgb
from skimage.filters import threshold_otsu

from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


@dataclass
class SampleCoords:
    """
    A sample's patch coordinate table and WSI slide metadata. Read from TRIDENT 
    metadata.
    """

    table       : pd.DataFrame  # tile_id, x, y, raw_file, mask_file, patch_name
    patch_size  : int
    slide_height: int
    slide_width : int


# =====| Stitch Vessel Predictions |============================================

def stitch_predictions(sample_id  : str,
                       predictions: dict[str, np.ndarray],
                       coords     : SampleCoords,
                       threshold  : float) -> tuple[np.ndarray, np.ndarray]:
    """
    Stitches per-patch predicted vessel content to the WSI slide level. Returns 
    a map and binary mask.
    """

    vessel_map, _ = place_patches(
        sample_id    = sample_id,
        coordinates  = coords.table,
        patch_size   = coords.patch_size,
        slide_height = coords.slide_height,
        slide_width  = coords.slide_width,
        get_patch    = lambda row: predictions.get(row['patch_name'])
    )
    vessel_mask = (vessel_map >= threshold).astype(np.uint8)

    return vessel_map, vessel_mask


def place_patches(sample_id     : str,
                  coordinates  : pd.DataFrame,
                  patch_size   : int,
                  slide_height : int,
                  slide_width  : int,
                  get_patch    : Callable[[pd.Series], np.ndarray | None]
                 ) -> np.ndarray: 
    """
    Places non-overlapping patches on to the full WSI slide canvas.
    """

    # Initialize a canvas for stitching and a boolean tracker
    slide     = np.zeros((slide_height, slide_width), dtype = np.float32)
    written   = np.zeros((slide_height, slide_width), dtype = bool)
    n_missing = 0

    # Place each row possessed in the coordinate table
    for _, row in coordinates.iterrows():
        patch = get_patch(row)
        if patch is None: n_missing += 1; continue

        y0, x0 = int(row['y']), int(row['x'])
        y1     = min(y0 + patch_size, slide_height)
        x1     = min(x0 + patch_size, slide_width)

        region = written[y0:y1, x0:x1]

        # No patches should overlap
        assert not region.any(), (
            f"Overlap detected for {sample_id} at patch {row['patch_name']}"
        )

        slide[y0:y1, x0:x1]   = patch[:, y1 - y0, x1 - x0]
        written[y0:y1, x0:x1] = True

    n = len(coordinates)
    logger.info(f"- Stitched {n - n_missing} / {n} patches")

    return slide

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
        tissue = load_tissue_mask(
            sample_id  = sample_id,
            src_dir    = mask_dir,
            tile_id    = row['tile_id'],
            patch_size = coordinates.patch_size
        )




    

    return



def stitch_tissue_mask():
    return

# [END]