# ==============================================================================
# Script:           cross_validation.py
# Purpose:          Patient-stratified K-Fold Cross Validation
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/22/2026
# ==============================================================================

import logging

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter

logger = logging.getLogger(__name__)
JOIN_KEY = ['sample_id', 'patch_name']

# =====| Main Functions |=======================================================

def assign_folds(manifest:   pd.DataFrame,
                 k:          int,
                 seed:       int,
                 n_pos_bins: int) -> pd.DataFrame:
    """
    Assign patient-level stratified k-fold indices to patches.

    StratifiedGroupKFold preserves CHIP status distribution across folds
    while enforcing patient-level grouping to prevent data leakage
    (Bradshaw et al., Radiol Artif Intell, 2023; scikit-learn docs).

    Join stratification is required because CHIP status alone does not control 
    for patient-level positive imbalance.

    Parameters
    ----------
    manifest : pd.DataFrame
        Patch manifest with 'sample_id', 'patch_name', 'chip_status'.
    k : int
        Number of folds.
    seed : int
        Random seed for reproducibility (Colliot et al., ML for Brain Disorders, 2023).
    n_pos_bins : int
        Number of quantile bins for patient_pos_count stratification.

    Returns
    -------
    pd.DataFrame
        Columns: sample_id, patch_name, fold (0-indexed int).
    """

    logger.info("=" * 50)
    logger.info("Step 03: Fold Assignment")
    logger.info(f"- Number of Folds (K): {k}")
    logger.info(f"- Seed: {seed}")
    logger.info("-" * 50)

    # Extract all patients and their CHIP status label
    patients = (manifest[['sample_id', 'chip_status', 'patient_pos_count']]
                .drop_duplicates('sample_id').copy())
    patients['fold'] = -1

    logger.info(f"Identified {len(patients)} patients for fold assignment")

    # Bin vessel-positive burden and combine with CHIP status
    patients['pos_bin'] = pd.qcut(
        x          = patients['patient_pos_count'],
        q          = n_pos_bins,
        labels     = False,
        duplicates = 'drop'
    )
    patients['strat_label'] = (
        patients['chip_status'].astype(str) + '_' + 
        patients['pos_bin'].astype(str)
    )
    
    # Stratify K folds at the patient level by CHIP status
    sgkf = StratifiedGroupKFold(
        n_splits     = k,
        shuffle      = True, 
        random_state = seed
    )

    # Assign all patients to their corresponding split
    fold_series = pd.Series(-1, index = patients.index)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(
        X      = patients,
        y      = patients['strat_label'],
        groups = patients['sample_id']
    )):
        fold_series.iloc[val_idx] = fold_idx

    patients['fold'] = fold_series

    # Post-hoc audit: ensure no fold is disproportionately over-under rep.
    fold_pos_counts = patients.groupby('fold')['patient_pos_count'].sum()
    expected = fold_pos_counts.sum() / k

    logger.info("Verifying all folds are proportionately represented:")

    for fold_idx, count in fold_pos_counts.items():
        ratio = count / expected
        logger.info(f"- Fold {fold_idx}: positive count = {count} "
                    f"({ratio:.2f}x expected)")
        if ratio > 2.0 or ratio < 0.5:
            logger.warning(f"Fold {fold_idx} deviates >2x from "
                           f"expected positive burden")


    logger.info(f"Successfully assigned all patients to {k} folds")
    logger.info("=" * 50)

    return (
        manifest[['sample_id', 'patch_name']]
        .assign(fold = manifest['sample_id'].map(
            patients.set_index('sample_id')['fold']
        ).astype(int))
    )
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assign_chip_labels(manifest: pd.DataFrame,
                       labels: pd.DataFrame,
                       positive_label: str,
                       negative_label: str) -> pd.DataFrame:
    """
    Merge CHIP status into patch manifest via substring matching. Maps
    positive and negative (CH/NC) to boolean labels.

    SampleID in chip_labels is a consistent substring of sample_id
    in manifest. Raises on any unmatched sample_id.

    Parameters
    ----------
    manifest : pd.DataFrame
        Patch manifest with 'sample_id' column.
    labels : pd.DataFrame
        CHIP labels with columns: SampleID, CHIP.

    Returns
    -------
    pd.DataFrame
        Manifest with 'chip_status' (int) populated.
    """

    logger.info("=" * 50)
    logger.info("Step 02: CHIP Label Assignment")
    logger.info(f"- Positive Label: {positive_label}")
    logger.info(f"- Negative Label: {negative_label}")
    logger.info("-" * 50)

    sample_ids = manifest['sample_id'].unique()
    chip_map = {}

    logger.info(f"Identified {len(sample_ids)} samples for mapping.")

    for sample_id in sample_ids:

        # Find all matching labels to the sample ID (substring)
        matches = labels[labels['SampleID'].apply(lambda sid: sid.lower() 
                                                  in sample_id.lower())]
        
        # All sample IDs must match to a CHIP status label
        if matches.empty: raise ValueError(f"No CHIP label found for "
                                           f"sample ID: {sample_id}")
        
        # Convert the labels to binary
        chip_label = matches.iloc[0]['CHIP']
        if chip_label == positive_label: chip_map[sample_id] = True
        elif chip_label == negative_label: chip_map[sample_id] = False
        else: raise ValueError(f"Unexpected CHIP value '{chip_label}' "
                               f"for sample ID: {sample_id}")

    manifest['chip_status'] = manifest['sample_id'].map(chip_map)

    logger.info("Successfully mapped all samples to CHIP status")

    bool_values = [bool(v) for v in chip_map.values()]
    counts = Counter(bool_values)

    n_chip = counts.get(True, 0)
    n_nochip = counts.get(False, 0)
        
    logger.info(f"Mapped {n_chip} patients to CHIP status")
    logger.info(f"Mapped {n_nochip} patients to Non-CHIP status")
    logger.info("=" * 50)

    return manifest
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def merge_fold_assignments(manifest: pd.DataFrame,
                           fold_assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fold assignments into patch manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Patch manifest with 'sample_id', 'patch_name'.
    fold_assignments : pd.DataFrame
        Output of assign_folds(); columns: sample_id, patch_name, fold.

    Returns
    -------
    pd.DataFrame
        Manifest with 'fold' (int) populated.
    """

    fold_lookup = fold_assignments.set_index(JOIN_KEY)

    manifest['fold'] = (manifest.set_index(JOIN_KEY).index.map(
        fold_lookup['fold'].to_dict()))

    return manifest

# [END]