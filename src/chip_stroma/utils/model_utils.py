# ==============================================================================
# Script:           model_utils.py
# Purpose:          Miscellaneous utility functions for model implementation
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import torch

import numpy as np

from scipy.ndimage import distance_transform_edt
from typing import cast




# =====| Helpers |==========================================================
    


def _edt(binary_img: np.ndarray) -> np.ndarray:
    """Wrapper to force correct static type through ambiguous scipy overload."""
    return cast(np.typing.NDArray[np.float64], 
                distance_transform_edt(binary_img))


def compute_dist_map(mask: torch.Tensor) -> torch.Tensor:
    """
    Signed distance transform of GT boundary, computed per-sample (B,H,W) on CPU.
    """
    
    mask_np = mask.cpu().numpy()
    dist_maps = np.stack([_edt(1 - m) - _edt(m) for m in mask_np])
    return torch.from_numpy(dist_maps).to(mask.device).float()


# [END]