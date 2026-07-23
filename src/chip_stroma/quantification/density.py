# ==============================================================================
# Script:           density.py
# Purpose:          Compute aSMA+ and vessel region area per patient
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

# src/quantify/vessel_density.py

def deconvolve_dab(patch_rgb: np.ndarray) -> np.ndarray:
    """Color deconvolution to isolate DAB channel (αSMA stain)."""
    ...

def otsu_vessel_area(dab_channel: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Otsu threshold within predicted vessel regions -> vessel area fraction."""
    ...

def per_patient_density(patch_manifest: pd.DataFrame, pred_masks_dir: str) -> pd.DataFrame:
    """Aggregate patch-level vessel area -> per-patient density score.
    Joint match on (sample_id, patch_name) per resolved patch identity bug.
    """
    ...

