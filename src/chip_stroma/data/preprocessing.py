# ==============================================================================
# Script:           paths.yaml
# Purpose:          aSMA stain and mask preprocessing
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import logging
import torchstain
import cv2

import pandas as pd
import numpy as np

from PIL import Image, PngImagePlugin
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from tiatoolbox.tools.stainnorm import VahadaneNormalizer, ReinhardNormalizer
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from skimage.filters import threshold_otsu, gaussian
from skimage.color import separate_stains, hdx_from_rgb
from skimage.morphology import (
    remove_small_objects, 
    remove_small_holes, 
    dilation,
    closing,
    opening,
    disk
)

logger = logging.getLogger(__name__)
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)  # 100MB

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

# =====| Orchestrational Wrappers |=============================================

def convert_vessel_masks(manifest: pd.DataFrame,
                         src_mask_dir: Path, 
                         dst_mask_dir: Path,
                         n_workers: int | None = None) -> pd.DataFrame:
    """
    Populate 'vessel_mask' and 'has_vessel' fields in the patch manifest.
    Copies vessel annotation mask patches to the destination directory, 
    converting its name from '_mask.png' to '_vessel_mask.png'.

    For each patch, resolves the corresponding vessel annotation mask at:
        src_mask_dir / original_id / patch_name.replace('_patch.png', '_mask.
        png')

    Parameters
    ----------
    manifest : pd.DataFrame
        Patch manifest with columns: 'original_id', 'patch_name'.
    src_mask_dir : Path
        Root directory containing per-sample mask subdirectories.

    Returns
    -------
    pd.DataFrame
        Report with 'vessel_mask' (str | None) and 'has_vessel' (bool) 
        populated.
    """

    logger.info("=" * 50)
    logger.info("Step 09: Vessel Mask Conversion")
    logger.info(f"- Source Vessel Mask Directory: {src_mask_dir}")
    logger.info(f"- Destination Vessel Mask Directory: {dst_mask_dir}")
    logger.info("-" * 50)

    # Convert the manifest to lightweight row objects for parallelization
    rows = [(str(row[0]), str(row[1]), str(row[2])) for row in 
            (manifest[['original_id', 'sample_id', 'patch_name']]
             .itertuples(index = False, name = None))]
    
    # Define workers for parallelization
    worker_fn = partial(
        vessel_worker, 
        src_mask_dir = src_mask_dir, 
        dst_mask_dir = dst_mask_dir
    )

    # Detect vessel annotation content in all patches
    report = []
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for r in tqdm(pool.map(worker_fn, rows, chunksize = 20),
                      total = len(rows)):
            report.append(r)
    report = pd.DataFrame.from_records(report)

    # Compute per-sample positive patch counts and merge into report
    pos_counts = (report[report['vessel_pixel_count'] > 0]
                  .groupby('sample_id').size().rename('patient_pos_count'))
    
    report = (report.join(pos_counts, on = 'sample_id')
              .fillna({'patient_pos_count': 0}))
    report['patient_pos_count'] = report['patient_pos_count'].astype(int)

    n_total  = len(report)
    n_passed = int(report["has_vessel"].sum())

    logger.info(f"Evaluated {n_total} total patches: {n_passed} patches "
                f"have vessel annotation content "
                f"({100 * n_passed / n_total:.1f}%)")
    logger.info("=" * 50)

    return report

    
def apply_tissue_filter(src_patch_dir: Path,
                        dst_mask_dir: Path,
                        manifest: pd.DataFrame,
                        tissue_threshold: float = 0.5,
                        gaussian_sigma: float = 1.0,
                        min_region_size: int = 500,
                        morph_disk_radius: int = 3,
                        n_workers: int | None = None) -> pd.DataFrame:
    """
    Orchestration wrapper for tissue detection.

    Iterates over all patches in the manifest, calls detect_tissue on each,
    and updates the manifest with the result. Does not perform any I/O
    beyond reading patches.

    Parameters
    ----------
    src_patch_dir : Path
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
        Report updated with 'passes_tissue' and 'include' columns.
    """

    logger.info("=" * 50)
    logger.info("Step 05: Tissue Detection")
    logger.info(f"- Source Patch Directory: {src_patch_dir}")
    logger.info(f"- Destination Tissue Mask Directory: {dst_mask_dir}")
    logger.info(f"- Tissue Threshold: {tissue_threshold}")
    logger.info(f"- Gaussian Sigma: {gaussian_sigma}")
    logger.info(f"- Minimum Region Size: {min_region_size}")
    logger.info(f"- Morphological Disk Radius: {morph_disk_radius}")
    logger.info("-" * 50)

    # Convert the manifest to lightweight row objects for parallelization
    rows = [(str(row[0]), str(row[1]), str(row[2])) for row in 
            (manifest[['original_id', 'sample_id', 'patch_name']]
             .itertuples(index = False, name = None))]

    # Define workers for parallelization
    worker_fn = partial(
        tissue_worker,
        src_patch_dir = src_patch_dir,
        dst_mask_dir = dst_mask_dir,
        tissue_threshold  = tissue_threshold,
        gaussian_sigma    = gaussian_sigma,
        min_region_size   = min_region_size,
        morph_disk_radius = morph_disk_radius
    )

    logger.info(f"Initialized {n_workers} workers for parallelization")
    logger.info("Beginning tissue detection across all workers")

    # Detect tissue content in all patches
    report = []
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for r in tqdm(pool.map(worker_fn, rows, chunksize = 20),
                      total = len(rows)):
            report.append(r)
    report = pd.DataFrame.from_records(report)

    n_passed        = int(report["passes_tissue"].sum())
    n_total         = len(report)
    fails_gray      = int(report["gray_fail"].sum())
    fails_threshold = int(report["threshold_fail"].sum())

    logger.info(f"Evaluated {n_total} total patches: ")
    logger.info(f"- Passed Tissue Filter: {n_passed} patches "
                f"({100 * n_passed / n_total:.1f}%)")
    logger.info(f"- Failed Gray Check: {fails_gray} patches "
                f"({100 * fails_gray / n_total:.1f}%)")
    logger.info(f"- Failed Threshold: {fails_threshold} patches "
                f"({100 * fails_threshold / n_total:.1f}%)")
    logger.info("=" * 50)
    
    return report


def apply_artifact_filter(src_dir: Path,
                          manifest: pd.DataFrame,
                          blur_threshold: float = 100.0,
                          dark_pixel_threshold: int = 50,
                          dark_pixel_ratio: float = 0.1,
                          pen_pixel_ratio: float = 0.05,
                          n_workers: int | None = None) -> pd.DataFrame:
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
        Report updated with 'passes_artifact' and 'include' columns.
    """

    logger.info("=" * 50)
    logger.info("Step 06: Artifact Detection")
    logger.info(f"- Blur Threshold: {blur_threshold}")
    logger.info(f"- Dark Pixel Threshold: {dark_pixel_threshold}")
    logger.info(f"- Dark Pixel Ratio: {dark_pixel_ratio}")
    logger.info(f"- Pen Pixel Ratio: {pen_pixel_ratio}")
    logger.info("-" * 50)

    # Convert the manifest to lightweight row objects for parallelization
    rows = [(str(row[0]), str(row[1]), str(row[2])) for row in 
            (manifest[['original_id', 'sample_id', 'patch_name']]
             .itertuples(index = False, name = None))]

    # Define workers for parallelization
    worker_fn = partial(
        artifact_worker,
        src_dir              = src_dir,
        blur_threshold       = blur_threshold,
        dark_pixel_threshold = dark_pixel_threshold,
        dark_pixel_ratio     = dark_pixel_ratio,
        pen_pixel_ratio      = pen_pixel_ratio
    )

    logger.info(f"Initialized {n_workers} workers for parallelization")
    logger.info("Beginning artifact detection across all workers")

    # Detect artifacts in all patches
    report = []
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for r in tqdm(pool.map(worker_fn, rows, chunksize = 20),
                      total = len(rows)):
            report.append(r)
    report = pd.DataFrame.from_records(report)

    n_passed = int(report['passes_artifact'].sum())
    n_total = len(report)

    logger.info(f"Evaluated {n_total} total patches: {n_passed} patches "
                f"passed the artifact detection filter "
                f"({100 * n_passed / n_total:.1f}%)")
    logger.info("=" * 50)
    
    return report


def normalize_patches(included_patches: pd.DataFrame,
                      normalizer,
                      method: str,
                      src_patch_dir: Path,
                      dst_patch_dir: Path,
                      n_workers: int | None = None) -> None:
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
        'sample_id', 'original_id', and 'patch_name' columns.
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
    logger.info("Step 08: Normalization")
    logger.info(f"- Source Patch Directory: {src_patch_dir}")
    logger.info(f"- Destination Patch Directory: {dst_patch_dir}")
    logger.info(f"- Method: {method}")
    logger.info("-" * 50)

    # Convert the manifest to lightweight row objects for parallelization
    rows = [(str(row[0]), str(row[1]), str(row[2])) for row in 
            (included_patches[['original_id', 'sample_id', 'patch_name']]
             .itertuples(index = False, name = None))]

    # Define workers for parallelization
    worker_fn = partial(
        normalize_worker,
        normalizer    = normalizer,
        method        = method,
        src_patch_dir = src_patch_dir,
        dst_patch_dir = dst_patch_dir
    )

    logger.info(f"Initialized {n_workers} workers for parallelization")
    logger.info("Beginning normalization across all workers")

    # Normalize all patches
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for _ in tqdm(pool.map(worker_fn, rows, chunksize = 20),
                      total = len(rows)):
            pass

    logger.info(f"Normalized {len(included_patches)} patches")
    logger.info("All masks moved to the corresponding destination directory")
    logger.info("=" * 50)
    
    return None


# =====| Multi-Threading Helpers |==============================================

def vessel_worker(row: tuple[str, str, str], 
                  src_mask_dir: Path,
                  dst_mask_dir: Path) -> dict:
    """Processes a single patch for vessel annotation content."""

    original_id, sample_id, patch_name = row
    src_mask_name = patch_name.replace("_raw.png", "_mask.png")
    dst_mask_name = patch_name.replace("_raw.png", "_vessel_mask.png")

    src_mask_path = src_mask_dir / original_id / src_mask_name
    dst_mask_path = dst_mask_dir / sample_id / dst_mask_name
    dst_mask_path.parent.mkdir(parents = True, exist_ok = True)
    
    if src_mask_path.exists():

        # Scale the binary mask to 255 (grayscale) to match tissue mask
        with Image.open(src_mask_path) as mask: mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(mask, mode = 'L').save(dst_mask_path)

        vessel_pixel_count = int(np.count_nonzero(mask))
        return {
            'sample_id':          sample_id,   # Join Key
            'patch_name':         patch_name,  # Join Key
            'vessel_mask':        dst_mask_name,
            'has_vessel':         vessel_pixel_count > 0,
            'vessel_pixel_count': vessel_pixel_count
        }

    return {
        'sample_id':          sample_id,
        'patch_name':         patch_name,
        'vessel_mask':        None,
        'has_vessel':         False,
        'vessel_pixel_count': 0
    }


def tissue_worker(row:               tuple[str, str, str],
                  src_patch_dir:     Path,
                  dst_mask_dir:      Path,
                  tissue_threshold:  float = 0.5,
                  gaussian_sigma:    float = 1.0,
                  min_region_size:   int = 500,
                  morph_disk_radius: int = 3) -> dict:
    """Processes a single patch for tissue detection."""

    original_id, sample_id, patch_name = row

    # Build the raw path from the manifest using the original sample ID
    patch_path = src_patch_dir / original_id / patch_name
    with Image.open(patch_path) as img: patch = np.array(img.convert("RGB"))

    # Evaluate if the patch contains sufficient tissue content
    mask, passed, tissue_ratio = detect_tissue(
        patch             = patch,
        tissue_threshold  = tissue_threshold,
        gaussian_sigma    = gaussian_sigma,
        min_region_size   = min_region_size,
        morph_disk_radius = morph_disk_radius
    )

    # If the patch passes, save its corresponding tissue mask
    if mask is not None:
        mask_name = patch_name.replace('_raw.png', '_tissue_mask.png')
        dst_path = dst_mask_dir / sample_id / mask_name
        dst_path.parent.mkdir(parents = True, exist_ok = True)
        Image.fromarray((mask * 255).astype(np.uint8), 
                        mode = 'L').save(dst_path)
    else: mask_name = None
        
    # Clean up intermediates to avoid memory overhead
    del patch
    del mask
        
    return {
        "sample_id":      sample_id,    # Join Key
        "patch_name":     patch_name,   # Join Key
        "passes_tissue":  passed,
        "tissue_mask":    mask_name,
        "tissue_ratio":   tissue_ratio,
        "gray_fail":      tissue_ratio == -1.0,
        "threshold_fail": (tissue_ratio != -1.0) and (not passed)
    }


def artifact_worker(row: tuple[str, str, str],
                    src_dir: Path,
                    blur_threshold: float = 100.0,
                    dark_pixel_threshold: int = 50,
                    dark_pixel_ratio: float = 0.1,
                    pen_pixel_ratio: float = 0.05) -> dict:
    """Processes a single patch for artifact detection."""

    original_id, sample_id, patch_name = row

    # Build the raw path from the manifest using the original sample ID
    patch_path = src_dir / original_id / patch_name
    with Image.open(patch_path) as img: patch = np.array(img.convert("RGB"))

    # Evaluate if the patch contains sufficient tissue content
    passed, lap_variance, dark_ratio, pen_ratio = detect_artifacts(
        patch                = patch,
        blur_threshold       = blur_threshold,
        dark_pixel_threshold = dark_pixel_threshold,
        dark_pixel_ratio     = dark_pixel_ratio,
        pen_pixel_ratio      = pen_pixel_ratio
    )

    del patch

    return {
        "sample_id":       sample_id,   # Join key
        "patch_name":      patch_name,  # Join key
        "passes_artifact": passed,
        "lap_variance":    lap_variance,
        "dark_ratio":      dark_ratio,
        "pen_ratio":       pen_ratio
    }


def normalize_worker(row: tuple[str, str, str],
                     normalizer,
                     method: str,
                     src_patch_dir: Path,
                     dst_patch_dir: Path) -> None:
    """Processes a single patch for normalization."""

    original_id, sample_id, patch_name = row

    # Build the source and destination paths based on sanitized names
    src_patch_path = src_patch_dir / original_id / patch_name
    dst_patch_path = dst_patch_dir / sample_id / patch_name
    dst_patch_path.parent.mkdir(parents = True, exist_ok = True)

    with Image.open(src_patch_path) as img: patch = np.array(img.convert('RGB'))
    normalized = normalize_patch(patch, normalizer, method)
    Image.fromarray(normalized, mode = 'RGB').save(dst_patch_path)

    del patch
    del normalized

    return 

# =====| Preprocessing Helpers |================================================

def detect_tissue(patch: np.ndarray,
                  tissue_threshold: float = 0.5,
                  gaussian_sigma: float = 1.0,
                  min_region_size: int = 500,
                  morph_disk_radius: int = 3
                  ) -> tuple[np.ndarray | None, bool, float]:
    """
    Detect tissue regions in an H-DAB IHC patch.

    Uses stain deconvolution (Ruifrok & Johnston, 2001) to separate
    haematoxylin and DAB channels, uses only haematoxylin for a tissue mask:
      1. Stain deconvolution via skimage hdx_from_rgb (H-DAB matrix)
      2. Gaussian smoothing on each channel
      3. Otsu thresholding on each channel independently
      4. Union mask: tissue present if haematoxylin OR DAB signal detected
      5. Morphological cleanup — remove small objects and fill small holes
      6. Reject patch if tissue ratio is below threshold

    Rationale for using both channels:
    - Haematoxylin (blue): marks all nucleated cells as a stain-agnostic tissue 
                           indicator
    - DAB (brown): marks αSMA+ cells — vessel/fibroblast signal
    - DAB-only detection would bias against patches with low αSMA expression,
      confounding CHIP vs non-CHIP comparison
    - Union mask ensures tissue-containing patches with low marker expression
      are retained

    Parameters
    ----------
    patch : np.ndarray
        RGB patch of shape (H, W, 3), dtype uint8.
    tissue_threshold : float
        Minimum fraction of pixels classified as tissue for the patch to pass.
        Default 0.5, consistent with CLAM (Lu et al., 2021).
    gaussian_sigma : float
        Sigma for Gaussian smoothing applied before Otsu thresholding.
        Default 1.0, consistent with HistomicsTK.
    min_region_size : int
        Minimum connected component size in pixels to retain.
        Default 500.
    morph_disk_radius : int
        Radius of the morphological structuring element for opening and closing.
        Default 3.

    Returns
    -------
    tuple[bool, float]
        (passed, tissue_ratio). tissue_ratio is -1.0 if patch fails early.

    References
    ----------
    - Ruifrok & Johnston (2001). Quantification of histochemical staining
      by color deconvolution. Anal Quant Cytol Histol.
    - Lu et al. (2021). Data-efficient and weakly supervised computational
      pathology on whole-slide images. Nature Biomedical Engineering. (CLAM)
    - HistomicsTK: Gaussian smoothing + Otsu thresholding for tissue detection.
    """

    # Validate the input format of the patch itself
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected RGB patch of shape (H, W, 3), "
                         f"got {patch.shape}")
    
    # Gray pre-rejection for mainly background patches
    R = patch[:, :, 0].astype(float)
    G = patch[:, :, 1].astype(float)
    B = patch[:, :, 2].astype(float)
    gray_mask = ((np.abs(R - G) < 10) & (np.abs(G - B) < 10) 
                 & (np.abs(R - B) < 10))
    if gray_mask.mean() > 0.9: return (None, False, -1.0)
    
    # Stain deconvolution: separate blue from brown
    stains = separate_stains(patch / 255.0, hdx_from_rgb)
    haem_raw = np.clip(stains[:, :, 0], 0, None).astype(np.float32)
    dab_raw  = np.clip(stains[:, :, 1], 0, None).astype(np.float32)

    haem_mask = np.zeros(haem_raw.shape, dtype = bool)
    dab_mask  = np.zeros(dab_raw.shape, dtype = bool)

    # Scale the haematoxylin channel to [0-1]
    haem = ((haem_raw - haem_raw.min()) 
            / (haem_raw.max() - haem_raw.min() + 1e-8))
    dab  = ((dab_raw - dab_raw.min()) 
            / (dab_raw.max() - dab_raw.min() + 1e-8))
    
    # Gaussian smoothing before thresholding
    haem_smoothed = gaussian(haem, sigma = gaussian_sigma)
    dab_smoothed  = gaussian(dab, sigma = gaussian_sigma)

    # Compute Otsu threshold per channel to segment for tissue
    haem_mask = haem_smoothed > threshold_otsu(haem_smoothed)
    dab_mask = dab_smoothed > threshold_otsu(dab_smoothed)

    tissue_mask = haem_mask | dab_mask

    # Perform morphological cleanup upon the tissue mask
    tissue_mask = dilation(tissue_mask, disk(morph_disk_radius * 1.25))
    selem       = disk(morph_disk_radius)
    tissue_mask = opening(tissue_mask, selem)
    tissue_mask = closing(tissue_mask, selem)

    # Remove small connected components for area filtering
    tissue_mask = remove_small_objects(
        tissue_mask, 
        max_size = min_region_size - 1
    )
    tissue_mask = remove_small_holes(
        tissue_mask, 
        max_size = min_region_size - 1
    )
    
    # Apply a minimum tissue percentage to keep patches
    tissue_ratio = tissue_mask.mean()
    if tissue_ratio < tissue_threshold: 
        return (tissue_mask, False, tissue_ratio)

    return (tissue_mask, True, tissue_ratio)


def detect_artifacts(patch: np.ndarray,
                     blur_threshold: float = 100.0,
                     dark_pixel_threshold: int = 50,
                     dark_pixel_ratio: float = 0.1,
                     pen_pixel_ratio: float = 0.05) -> tuple[bool, float, float, float]:
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
    tuple[bool, int]
        The artifact coverage if the patch passes the tissue filter, -1.0 if it fails.

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
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    lap_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Detect tissue folds via dark pixels below the threshold
    dark_mask = (R < dark_pixel_threshold) & \
                (B < dark_pixel_threshold) & \
                (G < dark_pixel_threshold)
    dark_ratio = dark_mask.mean()

    # Compute an extreme channel dominance heuristic to detect pens/markers
    blue_pen_mask = (B > R * 1.5) & (B > G * 1.5)
    green_pen_mask = (G > R * 1.5) & (G > B * 1.5)
    pen_mask = blue_pen_mask | green_pen_mask
    pen_ratio = pen_mask.mean()

    if ((lap_variance < blur_threshold) or 
        (dark_ratio > dark_pixel_ratio) or
        (pen_ratio > pen_pixel_ratio)):
        return (False, lap_variance, dark_ratio, pen_ratio)

    return (True, lap_variance, dark_ratio, pen_ratio)


def normalize_patch(patch: np.ndarray, normalizer, method: str) -> np.ndarray:
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

    if method in {"vahadane"}: return normalizer.transform(patch)
    if method in {"reinhard"}: return normalizer.normalize(patch)

    patch_tensor = T(patch)
    result = normalizer.normalize(patch_tensor)

    if isinstance(result, tuple): result = result[0]
    if result.ndim == 4: result = result.squeeze(0)

    normalized = result.cpu().numpy().clip(0, 255).astype(np.uint8)

    return normalized


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
    logger.info("Step 07: Fit Normalizer")
    logger.info(f"- Reference Path: {reference_path}")
    logger.info(f"- Method: {method}")
    logger.info("-" * 50)

    normalizers = {
        "vahadane": VahadaneNormalizer,
        "reinhard" : ReinhardNormalizer,
        "macenko":  torchstain.normalizers.MacenkoNormalizer,
        "reinhard": torchstain.normalizers.ReinhardNormalizer
    }

    if method not in normalizers: 
        raise ValueError(f"Unknown normalization method: {method}. Choose from "
                         f"{list(normalizers.keys())}.")
    
    # Only Vahadane method does not need the array converted to Tensor
    reference = np.array(Image.open(reference_path).convert("RGB"))
    if method not in {"vahadane", "reinhard"}: reference = to_tensor(reference)

    normalizer = normalizers[method]()
    normalizer.fit(reference)

    logger.info("Successfully fit the normalizer to the reference patch")
    logger.info("=" * 50)

    return normalizer

# [END]