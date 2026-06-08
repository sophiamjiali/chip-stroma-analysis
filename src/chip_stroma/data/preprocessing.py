# ==============================================================================
# Script:           paths.yaml
# Purpose:          aSMA stain and mask preprocessing
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import logging

import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from skimage.color import rgb2lab
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology.binary import binary_closing, binary_opening
from skimage.morphology import (
    remove_small_objects, 
    remove_small_holes, 
    disk
)

logger = logging.getLogger(__name__)

# =====| Orchestrational Wrappers |=============================================

def apply_tissue_filter(src_dir: Path,
                        manifest: pd.DataFrame,
                        tissue_threshold: float = 0.5,
                        gaussian_sigma: float = 1.0,
                        min_region_size: int = 500,
                        morph_disk_radius: int = 3) -> pd.DataFrame:
    """
    Orchestration wrapper for tissue detection.

    Iterates over all patches in the manifest, calls detect_tissue on each,
    and updates the manifest with the result. Does not perform any I/O
    beyond reading patches.

    Parameters
    ----------
    src_dir : Path
        Root directory containing raw patches, structured as src_dir/{original_name}/*.png
    manifest : pd.DataFrame
        Patch manifest produced by build_patch_manifest.
    tissue_threshold : float
        Passed to detect_tissue.
    gaussian_sigma : float
        Passed to detect_tissue.
    min_region_size : int
        Passed to detect_tissue.
    morph_disk_radius : int
        Passed to detect_tissue.

    Returns
    -------
    pd.DataFrame
        Manifest updated with 'passes_tissue' and 'include' columns.
    """

    manifest = manifest.copy()
    passes_filter = []

    # Evaluate the tissue filter upon each patch available
    for _, row in tqdm(manifest.iterrows(), total = len(manifest), 
                       desc = "Tissue Detection"):
        
        # Build the raw path from the manifest using the original sample ID
        patch_path = src_dir / row['original_id'] / row['patch']
        patch = np.array(Image.open(patch_path).convert("RGB"))

        # Evaluate if the patch contains sufficient tissue content
        passed = detect_tissue(
            patch             = patch,
            tissue_threshold  = tissue_threshold,
            gaussian_sigma    = gaussian_sigma,
            min_region_size   = min_region_size,
            morph_disk_radius = morph_disk_radius
        )
        passes_filter.append(passed)

    # Update the patch manifest with the boolean flag
    manifest['passes_tissue'] = passes_filter
    manifest['include'] = manifest['passes_tissue']

    n_passed, n_total = sum(passes_filter), len(passes_filter)
    logger.info(f"Tissue detection: {n_passed}/{n_total} patches passed "
                f"({100 * n_passed / n_total:.1f}%)")
    
    return manifest


def apply_artifact_filter(src_dir: Path,
                          manifest: pd.DataFrame,
                          blur_threshold: float = 100.0,
                          dark_pixel_threshold: int = 50,
                          dark_pixel_ratio: float = 0.1,
                          pen_pixel_ratio: float = 0.05) -> pd.DataFrame:
    """
    Orchestration wrapper for tissue detection.

    Iterates over all patches in the manifest, calls detect_artifacts on each,
    and updates the manifest with the result. Does not perform any I/O
    beyond reading patches.

    Parameters
    ----------
    src_dir : Path
        Root directory containing raw patches, structured as src_dir/{original_name}/*.png
    manifest : pd.DataFrame
        Patch manifest produced by build_patch_manifest.
    blur_threshold : float
        Passed to detect_artifacts.
    dark_pixel_threshold : int
        Passed to detect_artifacts.
    dark_pixel_ratio : float
        Passed to detect_artifacts.
    pen_pixel_ratio : float
        Passed to detect_artifacts.

    Returns
    -------
    pd.DataFrame
        Manifest updated with 'passes_artifact' and 'include' columns.
    """

    manifest = manifest.copy()
    passes_filter = []

    # Evaluate the tissue filter upon each patch available
    for _, row in tqdm(manifest.iterrows(), total = len(manifest), 
                       desc = "Artifact Detection"):
        
        # Build the raw path from the manifest using the original sample ID
        patch_path = src_dir / row['original_id'] / row['patch']
        patch = np.array(Image.open(patch_path).convert("RGB"))

        # Evaluate if the patch contains sufficient tissue content
        passed = detect_artifacts(
            patch                = patch,
            blur_threshold       = blur_threshold,
            dark_pixel_threshold = dark_pixel_threshold,
            dark_pixel_ratio     = dark_pixel_ratio,
            pen_pixel_ratio      = pen_pixel_ratio
        )
        passes_filter.append(passed)

    # Update the patch manifest with the boolean flag
    manifest['passes_artifact'] = passes_filter
    mask = (manifest['included'] == True) & (manifest['passes_artifact']==False)
    manifest.loc[mask, 'included'] = False

    n_passed, n_total = sum(passes_filter), len(passes_filter)
    logger.info(f"Artifact detection: {n_passed}/{n_total} patches passed "
                f"({100 * n_passed / n_total:.1f}%)")
    
    return manifest


def normalize_patches():
    return


# =====| Preprocessing Helpers |================================================

def detect_tissue(patch: np.ndarray,
                  tissue_threshold: float = 0.5,
                  gaussian_sigma: float = 1.0,
                  min_region_size: int = 500,
                  morph_disk_radius: int = 3) -> bool:
    """
        Detect tissue regions in an IHC patch.
 
    Follows the standard computational pathology approach used in HistomicsTK
    and adopted in IHC studies (Nguyen et al.):
      1. Convert to LAB colour space
      2. Apply Gaussian smoothing to the B channel (blue-yellow axis)
      3. Otsu threshold to separate tissue from background
      4. Morphological cleanup — remove small objects and fill small holes
      5. Reject patch if tissue ratio is below threshold
 
    For IHC, the LAB B channel is preferred over HSV saturation:
    - HSV saturation is well-suited to H&E (strong hue contrast)
    - LAB B channel is more robust for IHC where background may retain
      residual chromogen colour, causing HSV saturation to misclassify
      background as tissue (Nguyen et al.; HistomicsTK)
 
    Parameters
    ----------
    patch : np.ndarray
        RGB patch of shape (H, W, 3), dtype uint8.
    tissue_threshold : float
        Minimum fraction of pixels classified as tissue for the patch to pass.
        Default 0.5, consistent with CLAM (Lu et al., 2021).
        Increase to 0.7-0.8 if many background-dominant patches are observed.
    gaussian_sigma : float
        Sigma for Gaussian smoothing applied before Otsu thresholding.
        Reduces sensitivity to noise. Default 1.0, consistent with HistomicsTK.
    min_region_size : int
        Minimum connected component size in pixels to retain after thresholding.
        Removes spurious small tissue detections. Default 500.
    morph_disk_radius : int
        Radius of the morphological structuring element for opening and closing.
        Default 3, consistent with standard pathology preprocessing.
 
    Returns
    -------
    bool
        True if the patch passes the tissue filter, false if it fails.
 
    References
    ----------
    - Lu et al. (2021). Data-efficient and weakly supervised computational
      pathology on whole-slide images. Nature Biomedical Engineering. (CLAM)
    - Nguyen et al. LAB B-channel Otsu for IHC tissue detection.
    - HistomicsTK: Gaussian smoothing + Otsu thresholding for tissue detection.
    """

    # Validate the input format of the patch itself
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected RGB patch of shape (H, W, 3), "
                         f"got {patch.shape}")
    
    # Convert to LAB; B channel (blue-yellow) more robust for IHC
    lab = rgb2lab(patch)
    channel = lab[:, :, 2]

    # Gaussian smoothing before thresholding
    smoothed = gaussian(channel, sigma = gaussian_sigma)

    # Compute Otsu threshold to segment for tissue
    try: threshold = threshold_otsu(smoothed)
    except ValueError: return False

    tissue_mask = smoothed > threshold

    # Perform morphological cleanup upon the tissue mask
    selem = disk(morph_disk_radius)
    tissue_mask = binary_opening(tissue_mask, selem)
    tissue_mask = binary_closing(tissue_mask, selem)

    # Remove small connected components for area filtering
    tissue_mask = remove_small_objects(tissue_mask, min_size = min_region_size)
    tissue_mask = remove_small_holes(tissue_mask, 
                                     area_threshold = min_region_size)
    
    # Apply a minimum tissue percentage to keep patches
    tissue_ratio = tissue_mask.mean()
    if tissue_ratio < tissue_threshold: return False

    return True


def detect_artifacts(patch: np.ndarray,
                     blur_threshold: float = 100.0,
                     dark_pixel_threshold: int = 50,
                     dark_pixel_ratio: float = 0.1,
                     pen_pixel_ratio: float = 0.05) -> bool:
    """
    Detect and reject patches containing common histopathology artifacts.

    Detects three artifact classes using classical image processing:

    1. Blur (out-of-focus) — variance of the Laplacian operator on the
       grayscale patch. Low variance indicates low edge content, i.e. blur.
       Threshold of 100 is the standard in WSI QC pipelines
       (PAM50 study; HistoQC).

    2. Tissue folds — detected as abnormally dark pixels where all RGB
       channels fall below a threshold. Folded tissue compresses and darkens
       the chromogen signal (HistoQC; Kanwal et al., 2024).

    3. Pen/marker artifacts — detected via extreme channel dominance:
       blue markers (B > R+G) and green markers (G > R+B), consistent
       with HistoQC's channel-dominance heuristic (Janowczyk et al.).

    Note: Air bubbles and blood artifacts are not implemented here as
    they are less common in bone marrow IHC and are better handled
    by a supervised classifier if needed (Kanwal et al., 2024).

    Parameters
    ----------
    patch : np.ndarray
        RGB patch of shape (H, W, 3), dtype uint8.
    blur_threshold : float
        Minimum Laplacian variance for a patch to be considered in-focus.
        Patches below this threshold are rejected. Default 100, consistent
        with standard WSI QC pipelines (PAM50 study).
    dark_pixel_threshold : int
        Maximum pixel intensity (per channel) to classify a pixel as dark.
        Pixels where R, G, B are all below this value are flagged as
        potential fold pixels. Default 50.
    dark_pixel_ratio : float
        Maximum fraction of dark pixels tolerated before rejecting the patch.
        Default 0.1 (10%).
    pen_pixel_ratio : float
        Maximum fraction of pen/marker pixels tolerated before rejecting.
        Default 0.05 (5%).

    Returns
    -------
    bool
        True if the patch passes the tissue filter, false if it fails.

    References
    ----------
    - Kanwal et al. (2024). Equipping computational pathology systems with
      artifact processing pipelines. BMC Medical Informatics.
    - Janowczyk et al. HistoQC: An Open-Source Quality Control Tool for
      Digital Pathology Slides.
    - PAM50 study: blur threshold of 100 for Laplacian variance in WSI QC.
    """

    # Validate the input format of the patch itself
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected RGB patch of shape (H, W, 3), "
                         f"got {patch.shape}")
    
    # Extract each colour channel from the validated patch
    R = patch[:, :, 0].astype(np.float32)
    G = patch[:, :, 1].astype(np.float32)
    B = patch[:, :, 2].astype(np.float32)

    # Apply blurring via variance of Laplacian on grayscale
    gray = np.mean(patch, axis = 2).astype(np.uint8)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype = np.float32)
    lap_response = np.abs((np.convolve(gray.rave(), laplacian.ravel(), 
                                       mode = "same").reshape(gray.shape)))
    laplacian_var = lap_response.var()

    if laplacian_var < blur_threshold: return False

    # Detect tissue folds via dark pixels below the threshold
    dark_mask = (R < dark_pixel_threshold) & \
                (B < dark_pixel_threshold) & \
                (G < dark_pixel_threshold)
    dark_ratio = dark_mask.mean()

    if dark_ratio > dark_pixel_ratio: return False

    # Compute an extreme channel dominance heuristic to detect pens/markers
    blue_pen_mask = B > (R + G)
    green_pen_mask = G > (R + B)
    pen_mask = blue_pen_mask | green_pen_mask
    pen_ratio = pen_mask.mean()

    if pen_ratio > pen_pixel_ratio: return False

    return True
    
# [END]