# ==============================================================================
# Script:           transforms.py
# Purpose:          Stain normalization and augmentation
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import albumentations as A


def get_train_transforms() -> A.Compose:
    """
    Joint spatial and stain augmentation pipeline for training patches.

    Geometric augmentation follows Ronneberger et al. (MICCAI, 2015).
    Elastic deformation follows the original U-Net formulation for
    biomedical image segmentation.
    Stain jitter via HSV perturbation follows Tellez et al. (IEEE TMI,
    2019), who demonstrate stain augmentation outperforms normalization
    alone for IHC generalization.
    Gaussian blur and noise follow Wagner et al. (Mod Pathol, 2023) for
    histopathology robustness.

    Returns
    -------
    A.Compose
        Albumentations pipeline with joint patch + vessel mask transforms.
    """

    return A.Compose([

        # Geometric augmentation (Ronneberger et al., MICCAI, 2015)
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.RandomRotate90(p = 0.5),

        # Elastic deformation (Ronneberger et al., MICCAI, 2015)
        A.ElasticTransform(
            alpha = 120,
            sigma = 120 * 0.05, # OG UNet parameters for biomedical segmentation
            p = 0.3
        ),

        # Stain jitter via HSV perturbation for IHC generalization
        # (Tellez et al., IEEE TMI, 2019)
        A.HueSaturationValue(
            hue_shift_limit = 10,
            sat_shift_limit = 20,
            val_shift_limit = 10,
            p = 0.5,
        ),

        # Gaussian Blur & Noise (Wagner et al., Mod Pathol, 2023)
        A.GaussianBlur(blur_limit = (3, 5), p = 0.2),
        A.GaussNoise(p = 0.2)
    ])


def get_val_transforms() -> A.Compose:
    """
    Validation transform pipeline — no augmentation.

    Validation set must reflect true data distribution without augmentation
    to produce unbiased performance estimates (Varoquaux & Cheplygina,
    NeuroImage, 2022; Bradshaw et al., Radiol Artif Intell, 2023).

    Returns
    -------
    A.Compose
        Identity pipeline for patch + vessel mask.
    """

    return A.Compose([])

# [END]