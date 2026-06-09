# ==============================================================================
# Script:           paths.yaml
# Purpose:          aSMA stain and mask preprocessing
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import shutil
import logging
import torchstain
import tiatoolbox

import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
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

    logger.info("=" * 50)
    logger.info("Step 04: Tissue Detection")
    logger.info(f"- Tissue Threshold: {tissue_threshold}")
    logger.info(f"- Gaussian Sigma: {gaussian_sigma}")
    logger.info(f"- Minimum Region Size: {min_region_size}")
    logger.info(f"- Morphological Disk Radius: {morph_disk_radius}")
    logger.info("-" * 50)

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

    logger.info(f"Evaluated {n_total} total patches: {n_passed} patches "
                f"passed the tissue detection filter "
                f"({100 * n_passed / n_total:.1f}%)")
    logger.info("=" * 50)
    
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

    logger.info("=" * 50)
    logger.info("Step 05: Artifact Detection")
    logger.info(f"- Blur Threshold: {blur_threshold}")
    logger.info(f"- Dark Pixel Threshold: {dark_pixel_threshold}")
    logger.info(f"- Dark Pixel Ratio: {dark_pixel_ratio}")
    logger.info(f"- Pen Pixel Ratio: {pen_pixel_ratio}")
    logger.info("-" * 50)

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

    logger.info(f"Evaluated {n_total} total patches: {n_passed} patches "
                f"passed the artifact detection filter "
                f"({100 * n_passed / n_total:.1f}%)")
    logger.info("=" * 50)
    
    return manifest


def normalize_patches(included_patches: pd.DataFrame,
                      normalizer,
                      src_patch_dir: Path,
                      src_mask_dir: Path,
                      dst_patch_dir: Path,
                      dst_mask_dir: Path) -> None:
    """
    Orchestration wrapper for stain normalization.

    Iterates over included patches, normalizes each, and saves to disk.
    Copies corresponding masks in lockstep with normalized patches.
    Operates only on patches where include=True — filtering is assumed
    to be complete before this function is called.

    Parameters
    ----------
    included_patches : pd.DataFrame
        Subset of manifest where include=True, containing at minimum
        'sample_id', 'original_name', and 'patch' columns.
    normalizer : fitted torchstain normalizer
        Pre-fitted normalizer instance. Must be fitted before calling.
    src_patch_dir : Path
        Root directory of raw patches: src_patch_dir/{original_name}/{patch}.
    src_mask_dir : Path
        Root directory of raw masks: src_mask_dir/{original_name}/{mask}.
    dst_patch_dir : Path
        Root directory for normalized patches: dst_patch_dir/{sample_id}/{patch}.
    dst_mask_dir : Path
        Root directory for copied masks: dst_mask_dir/{sample_id}/{mask}.
    """

    logger.info("=" * 50)
    logger.info("Step 07: Normalization")
    logger.info(f"- Source Patch Directory: {src_patch_dir}")
    logger.info(f"- Source Mask Directory: {src_mask_dir}")
    logger.info(f"- Destination Patch Directory: {dst_patch_dir}")
    logger.info(f"- Destination Mask Directory: {dst_mask_dir}")
    logger.info("-" * 50)

    # Normalize and save each patch under its sanitized sample ID and name
    for _, row in tqdm(included_patches.iterrows(), 
                       total = len(included_patches),
                       desc = "Patch Normalization:"):
        
        # Build the source and destination paths based on sanitized names
        src_patch_path = src_patch_dir / row['original_id'] / row['patch']
        dst_patch_path = dst_patch_dir / row['sample_id'] / row['patch']
        dst_patch_path.parent.mkdir(parents = True, exist_ok = True)

        patch = np.array(Image.open(src_patch_path).convert("RGB"))
        normalized = normalize_patch(patch, normalizer)
        Image.fromarray(normalized).save(dst_patch_path)

        # Copy the mask in lockstep
        mask_name = row['patch'].replace("_raw.png", "_mask.png")
        src_mask_path = src_mask_dir / row['original_id'] / mask_name
        dst_mask_path = dst_mask_dir / row['sample_id'] / mask_name
        dst_mask_path.parent.mkdir(parents = True, exist_ok = True)

        if src_mask_path.exists(): shutil.copy(src_mask_path, dst_mask_dir)
        else: print(f"Warning: missing mask for {row['patch']}")

    logger.info(f"Normalized {len(included_patches)} patches")
    logger.info("All masks moved to the corresponding destination directory")
    logger.info("=" * 50)
    
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


def normalize_patch(patch: np.ndarray, normalizer) -> np.ndarray:
    """
    Normalize a single RGB patch using a pre-fitted stain normalizer.

    Parameters
    ----------
    patch : np.ndarray
        RGB patch of shape (H, W, 3), dtype uint8.
    normalizer : fitted torchstain normalizer
        Pre-fitted normalizer instance (e.g. VahadaneNormalizer).
        Must be fitted on a reference patch before being passed in.

    Returns
    -------
    np.ndarray
        Normalized RGB patch of shape (H, W, 3), dtype uint8.
    """

    patch_tensor = to_tensor(patch)
    normalized = normalizer.normalize(patch_tensor)
    return (normalized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def fit_normalizer(reference_path: Path, method: str = "vahadane"):
    """
    Fit a stain normalizer on a reference patch.

    Parameters
    ----------
    reference_path : Path
        Path to a representative reference patch — ideally selected
        and reviewed by a pathologist.
    method : str
        Normalization method. One of: "vahadane" (default), "macenko", "reinhard".
        Vahadane is preferred for IHC (Vahadane et al., 2016).

    Returns
    -------
    Fitted torchstain normalizer instance.
    """

    logger.info("=" * 50)
    logger.info("Step 06: Fit Normalizer")
    logger.info(f"- Reference Path: {reference_path}")
    logger.info(f"- Method: {method}")
    logger.info("-" * 50)

    normalizers = {
        "vahadane": tiatoolbox.tools.stainnorm.VahadaneNormalizer,
        "macenko":  torchstain.normalizers.MacenkoNormalizer,
        "reinhard": torchstain.normalizers.ReinhardNormalizer
    }

    if method not in normalizers: 
        raise ValueError(f"Unknown normalization method: {method}. Choose from "
                         f"{list(normalizers.keys())}.")
    
    reference = to_tensor(np.array(Image.open(reference_path).convert("RGB")))
    normalizer = normalizers[method]()
    normalizer.fit(reference)

    logger.info("Successfully fit the normalizer to the reference patch")
    logger.info("=" * 50)

    return normalizer

# [END]