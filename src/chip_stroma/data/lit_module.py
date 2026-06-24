# ==============================================================================
# Script:           lit_module.py
# Purpose:          Trainer entrypoint
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
#
# Notes:            Global aggregation level for Dice Score is used for rare
#                   classes where per-sample zero-denominator fallback values 
#                   distort macro averages; backgrounds are not included to 
#                   report foreground (vessel) content only
# ==============================================================================

import lightning.pytorch as pl
import torch.nn as nn

from torchmetrics.segmentation import MeanIoU, DiceScore

from chip_stroma.segmentation.loss import FocalTverskyLoss, MaskedDiceLoss


class VesselSegModule(pl.LightningModule):
    """
    Args:
        model:        SMP U-Net (activation=None, returns raw logits).
        lr:           Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        T_max:        Cosine annealing period — set equal to max_epochs.
        ftl_alpha:    FTL FP weight. Tune: lower = tolerate more FP.
        ftl_beta:     FTL FN weight. Tune: higher = penalise missed vessels more.
        ftl_gamma:    FTL focal exponent. Tune: lower = harder focus on positives.
        ftl_weight:   FTL contribution to composite loss; Dice = 1 - ftl_weight.
        smooth:       Laplace smoothing shared by FTL and Dice loss.
    """

    def __init__(self,
                 model:        nn.Module,
                 lr:           float = 1e-4,
                 weight_decay: float = 1e-2,
                 T_max:        int = 50,
                 ftl_alpha:    float = 0.3,
                 ftl_beta:     float = 0.7,
                 ftl_gamma:    float = 0.75,
                 ftl_weight:   float = 0.5,
                 smooth:       float = 1.0):
        
        super().__init__()
        self.save_hyperparameters(ignore = ['model'])
        self.model = model

        # Initialize Focal Tversky Loss function
        self.ftl = FocalTverskyLoss(
            alpha  = ftl_alpha,
            beta   = ftl_beta,
            gamma  = ftl_gamma,
            smooth = smooth
        )

        # Initialize masked Dice Loss function
        self.dice_loss = MaskedDiceLoss(smooth = smooth)

        # Dice Score is initialized with a global aggregation level
        self.val_dice = DiceScore(
            num_classes        = 2,
            include_background = False,
            aggregation_level  = "global",
            input_format       = "index"
        )

        # Initialize Mean IoU as one-hot input, converted in validation step
        self.val_iou = MeanIoU(num_classes = 2, input_format = "index")

# [END]