# ==============================================================================
# Script:           loss.py
# Purpose:          Loss function definitions
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
#
# Notes:            Loss: Focal Tversky Loss (FTL) + Dice on tissue pixels   
#
# References:
#  - Abraham & Khan (2019) ISBI — Focal Tversky Loss
#    https://doi.org/10.1109/ISBI.2019.8759329
#  - Yeung et al. (2022) Comput Med Imaging Graph — Unified Focal Loss
#    https://doi.org/10.1016/j.compmedimag.2021.102026
#  - Salehi et al. (2017) MLMI — Tversky Loss
#    https://doi.org/10.1007/978-3-319-67389-9_44
# ==============================================================================

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.segmentation import MeanIoU, DiceScore

class FocalTverskyLoss(nn.Module):
    """
    FTL = (1 - TI)^gamma
    TI  = TP / (TP + alpha*FP + beta*FN)
 
    beta > alpha penalises missed vessels (FN) more than FP, appropriate
    for ~125:1 background/vessel imbalance (Abraham & Khan, 2019).
    gamma < 1 applies focal modulation, up-weighting hard examples.
 
    Args:
        alpha:  FP weight. Default 0.3.
        beta:   FN weight. Default 0.7.
        gamma:  Focal exponent. Default 0.75.
        smooth: Laplace smoothing to avoid division by zero.
    """

    def __init__(self,
                 alpha:  float = 0.3,
                 beta:   float = 0.7,
                 gamma:  float = 0.75,
                 smooth: float = 1.0):
        
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self,
                logits:  torch.Tensor, # (B, 1, H, W) raw logits
                targets: torch.Tensor, # (B, 1, H, W) binary float
                tissue:  torch.Tensor  # (B, 1, H, W) binary float tissue mask
                ) -> torch.Tensor:
        
        # Restrict all terms to tissue pixels only
        probs   = torch.sigmoid(logits)
        probs   = (probs * tissue).view(-1)
        targets = (targets * tissue).view(-1)

        # Compute true positives, false positives, and false negatives
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        # Compute the Tversky Index using each rate and parameter
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return (1 - tversky_index) ** self.gamma
    

class MaskedDiceLoss(nn.Module):
    """
    Soft Dice loss restricted to tissue pixels.
    Complements FTL by optimising global region overlap (Yeung et al., 2022).
    """

    def __init__(self, smooth: float = 1.0): 
        super().__init__()
        self.smooth = smooth

    def forward(self,
                logits:  torch.Tensor, # (B, 1, H, W) raw logits
                targets: torch.Tensor, # (B, 1, H, W) binary float
                tissue:  torch.Tensor  # (B, 1, H, W) binary float tissue mask
                ) -> torch.Tensor:
        
        # Restrict all terms to tissue pixels only
        probs   = torch.sigmoid(logits)
        probs   = (probs * tissue).view(-1)
        targets = (targets * tissue).view(-1)

        # Compute the intersection as the loss value
        intersection = (probs * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
    
# [END]