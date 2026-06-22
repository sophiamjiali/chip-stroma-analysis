# ==============================================================================
# Script:           dataset.py
# Purpose:          PyTorch Dataset for patches and masks
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import torch

import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image, PngImagePlugin
from typing import Callable, Optional
from torch.utils.data import Dataset

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)  # 100MB


class VesselPatchDataset(Dataset):
    """
    Patch dataset for vessel segmentation in bone marrow IHC biopsies.

    Loads Vahadane-normalized RGB patches with binary vessel and tissue
    masks. Transforms injected externally per train/val split following
    established PyTorch augmentation patterns (Buslaev et al.,
    Information, 2020).

    Tissue masks applied in loss computation (not as model input) to
    restrict gradient updates to valid tissue regions, preventing
    background bias in severely class-imbalanced segmentation tasks
    (Ronneberger et al., MICCAI, 2015; Nair et al., PMC, 2020).

    Parameters
    ----------
    manifest : pd.DataFrame
        Pre-filtered patch manifest (include=True rows only).
    patch_dir : Path
        Directory of Vahadane-normalized patch PNGs.
    vessel_mask_dir : Path
        Directory of binary vessel annotation masks (0/255, mode L).
    tissue_mask_dir : Path
        Directory of binary tissue masks (0/255, mode L).
    transform : callable, optional
        Albumentations transform applied to patch and vessel mask jointly.
    """

    def __init__(self,
                 manifest:        pd.DataFrame,
                 patch_dir:       Path,
                 vessel_mask_dir: Path,
                 tissue_mask_dir: Path,
                 transform:       Optional[Callable] = None) -> None:
        
        self.records         = manifest.reset_index(drop = True)
        self.patch_dir       = patch_dir
        self.vessel_mask_dir = vessel_mask_dir
        self.tissue_mask_dir = tissue_mask_dir
        self.transform       = transform

    # =====| Class Overloads |==================================================
    
    def __getitem__(self, idx: int) -> dict:

        # Load the normalized RGB patch from the patch directory
        row = self.records.iloc[idx]
        patch_path = self.patch_dir / row['sample_id'] / row['patch_name']
        with Image.open(patch_path) as img: patch = np.array(img.convert('RGB'))

        # Load the binary masks (0/255 uint8, mode 'L')
        vessel_path = self.vessel_mask_dir / row['sample_id'] /row['patch_name']
        with Image.open(vessel_path) as img: vessel_mask = np.array(img)

        tissue_path = self.tissue_mask_dir / row['sample_id'] /row['patch_name']
        with Image.open(tissue_path) as img: tissue_mask = np.array(img)

        # Apply joint spatial transforms to patch and vessel masks (not tissue)
        if self.transform is not None:
            transformed = self.transform(image = patch, mask = vessel_mask)
            patch       = transformed['image']
            vessel_mask = transformed['mask']

        # Convert patch and masks to tensors: HWC -> CHW, normalize to [0., 1.]
        patch       = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        vessel_mask = torch.from_numpy(vessel_mask).unsqueeze(0).float() / 255.0
        tissue_mask = torch.from_numpy(tissue_mask).squeeze(0).float() / 255.0

        return {
            'patch':       patch,
            'vessel_mask': vessel_mask,
            'tissue_mask': tissue_mask,
            'has_vessel':  bool(row['has_vessel']),
            'sample_id':   str(row['sample_id'])
        }
    
    def __len__(self) -> int:
        return len(self.records)
    
# [END]