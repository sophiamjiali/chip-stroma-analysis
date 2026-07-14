# ==============================================================================
# Script:           model_utils.py
# Purpose:          Miscellaneous utility functions for model implementation
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import torch
import warnings

import numpy as np

from scipy.ndimage import distance_transform_edt
from typing import cast

# =====| Helpers |==========================================================

def _edt(binary_img: np.ndarray) -> np.ndarray:
    """Wrapper to force correct static type through ambiguous scipy overload."""
    return cast(np.typing.NDArray[np.float64], 
                distance_transform_edt(binary_img))


def compute_dist_map(mask: torch.Tensor, 
                     max_dist: float = 20.0) -> torch.Tensor:
    """
    Signed distance transform of GT boundary, computed per-sample (B,H,W) on CPU. Clamped to +/- max_dist to bound the boundary-loss penalty.
    """

    mask_np = mask.cpu().numpy()
    dist_maps = np.stack([_edt(1 - m) - _edt(m) for m in mask_np])
    dist_maps = np.clip(dist_maps, -max_dist, max_dist)
    return torch.from_numpy(dist_maps).to(mask.device).float()


# Macro reduction: nanmean within patient, then across patients
def macro(sample_dict):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        per_sample = [np.nanmean(v) for v in sample_dict.values()]
        means      = torch.tensor(np.nanmean(per_sample))
        std        = torch.tensor(np.nanstd(per_sample))
        return means, std


# [END]