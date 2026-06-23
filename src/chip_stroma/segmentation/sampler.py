# ==============================================================================
# Script:           sampler.py
# Purpose:          Patient-balanced positive oversampling sampler
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/21/2026
# ==============================================================================

import numpy as np
import pandas as pd

from torch.utils.data import Sampler


class PositiveWeightedSampler(Sampler):
    """
    Patient-balanced positive oversampling sampler for vessel segmentation.

    Samples each batch index with probability `pos_prob` from positives
    and `1 - pos_prob` from negatives. Within positives, patches are
    weighted by 1/n_positives_per_patient to prevent high-density patients
    from dominating training signal (h-BMO-18 contributes 21.6% of filtered
    positives without correction).

    Per-patient weighting follows established practice for imbalanced
    medical imaging datasets with inter-patient annotation variability
    (Buda et al., Neural Networks, 2018; Nair et al., PMC, 2020).
    Negative patches weighted uniformly.

    Parameters
    ----------
    manifest : pd.DataFrame
        Pre-filtered training fold manifest with 'has_vessel', 'sample_id'.
    pos_prob : float
        Probability of sampling a positive patch per draw (default: 0.5).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self,
                 manifest: pd.DataFrame,
                 pos_prob: float = 0.5,
                 seed:     int = 42) -> None:
        
        self.pos_prob = pos_prob
        self.rng      = np.random.default_rng(seed)
        self.n        = len(manifest)

        # Extract positive and negative patch indices
        manifest = manifest.reset_index(drop = True)
        pos_mask = manifest['has_vessel'].astype(bool)

        self.pos_idx = manifest.index[pos_mask].to_numpy()
        self.neg_idx = manifest.index[~pos_mask].to_numpy()

        # Per-patient inverse-frequency weights for positive patches
        pos_manifest     = manifest.loc[pos_mask, 'sample_id']
        patient_counts   = pos_manifest.value.counts()
        patient_weights  = 1.0 / patient_counts
        patch_weights    = pos_manifest.map(patient_weights).to_numpy()
        self.pos_weights = patch_weights / patch_weights.sum()

        # Protect against RNG edge cases for training fold split
        if len(self.pos_idx) == 0:
            raise ValueError("Training fold contains no positive patches.")
        if len(self.neg_idx) == 0:
            raise ValueError("Training fold contains no negative patches.")
        
    
    # =====| Class Overloads |==================================================

    def __iter__(self):

        # Randomly draw the ratio of positive and negative patches
        draw_positive = self.rng.random(self.n) < self.pos_prob
        n_pos = draw_positive.sum()
        n_neg = self.n - n_pos
        
        # Sample the patches from the dataset indices
        sampled_pos = self.rng.choice(
            self.pos_idx, 
            size    = n_pos, 
            p       = self.pos_weights, 
            replace = True
        )
        sampled_neg = self.rng.choice(
            self.neg_idx,
            size    = n_neg, 
            replace = True
        )

        # Shuffle the positive and negative patches for drawing
        indices = np.concatenate([sampled_pos, sampled_neg])
        self.rng.shuffle(indices)

        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.n

# [END]