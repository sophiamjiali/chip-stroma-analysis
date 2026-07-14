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

import torch.nn.functional as F

from surface_distance import (
    compute_surface_distances, 
    compute_surface_dice_at_tolerance
)


# =====| Tversky Loss |=========================================================

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


# =====| Boundary Loss |========================================================

def boundary_loss(pred_probs: torch.Tensor, 
                  dist_map:   torch.Tensor) -> torch.Tensor:
    """
    Kervadec et al. 2019 boundary loss: penalizes pred mass weighted by 
    distance from GT boundary.
    """
    return (pred_probs * dist_map).mean()

# =====| Dice Loss |============================================================

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
        return 1 - (2 * intersection) / (
            probs.sum() + targets.sum() + self.smooth
        )

def dice_score(pred, true, eps=1e-8):
    """Per-sample Dice; smooth in denominator only."""
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * true).sum(dim=dims)
    denom = pred.sum(dim=dims) + true.sum(dim=dims)
    dice = (2 * intersection) / (denom + eps)
    dice[denom == 0] = float("nan")
    return dice


def per_sample_dice(preds, 
                    target, 
                    num_classes        = 2, 
                    include_background = False, 
                    eps                = 1e-6) -> torch.Tensor:
    """
    Per-sample foreground Dice score; excludes samples with no 
    positive pixels in either prediction or target.
    """

    preds_oh  = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).bool()
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).bool()

    if not include_background:
        preds_oh, target_oh = preds_oh[:, 1:], target_oh[:, 1:]

    dims = tuple(range(2, preds_oh.ndim))

    intersection = (preds_oh & target_oh).sum(dim = dims + (1,)).float()
    union = (preds_oh.sum(dim = dims + (1,)).float() 
                + target_oh.sum(dim = dims + (1,)).float())
    
    # Exclude true-negative samples from aggregation
    dice = (2 * intersection + eps) / (union + eps)
    dice[union == 0] = float('nan')

    return dice


def per_sample_surface_dice(y_pred, y, tolerance, spacing_mm = (1.0, 1.0)):
    """One-shot NSD for a bathc of one-hot predictions/targets."""
    out = []
    for b in range(y_pred.shape[0]):
        sd = compute_surface_distances(
            y[b, 1].cpu().numpy().astype(bool),
            y_pred[b, 1].cpu().numpy().astype(bool),
            spacing_mm
        )
        out.append(compute_surface_dice_at_tolerance(sd, tolerance))
    return torch.tensor(out)
    
# [END]