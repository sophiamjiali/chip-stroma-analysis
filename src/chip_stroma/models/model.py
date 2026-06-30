# ==============================================================================
# Script:           model.py
# Purpose:          LightningModule wrapping SMP U-Net
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
#
# Notes:            Global aggregation level for Dice Score is used for rare
#                   classes where per-sample zero-denominator fallback values 
#                   distort macro averages; backgrounds are not included to 
#                   report foreground (vessel) content only
# ==============================================================================

import torch

import segmentation_models_pytorch as smp
import lightning.pytorch as pl
import torch.nn as nn

from torchmetrics.segmentation import MeanIoU, DiceScore
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from chip_stroma.models.loss import FocalTverskyLoss, MaskedDiceLoss


def build_model(encoder_name:     str = "resnet34",
                encoder_weights:  str = "imagenet",
                decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
                in_channels:      int = 3,
                out_classes:      int = 1) -> nn.Module:
    """
    Factory function for smp U-Net vessel segmentation model.

    ResNet34 encoder selected for optimal capacity/overfitting tradeoff
    at n=24 patients (Raghu et al., NeurIPS, 2019; Talo et al.,
    Comput Biol Med, 2019). ImageNet pretraining improves convergence
    on small medical imaging datasets (Chen et al., CVPR, 2021).

    Factory pattern follows established research pipeline conventions
    for config-driven model instantiation (Goyal et al., FAIR, 2022;
    Bouthillier et al., NeurIPS, 2021).

    Parameters
    ----------
    encoder_name : str
        smp encoder backbone. Default: 'resnet34'.
    encoder_weights : str
        Pretrained weights source. Default: 'imagenet'.
    decoder_channels : tuple[int, ...]
        Feature channels per decoder block, high to low resolution.
        Length must equal encoder depth. Default: (256, 128, 64, 32, 16).
    in_channels : int
        Input image channels. Default: 3 (RGB).
    out_classes : int
        Output segmentation classes. Default: 1 (binary vessel mask).

    Returns
    -------
    nn.Module
        Initialized smp U-Net with pretrained encoder.
    """

    # Return the pretrained ImageNet encoder; loads pre-downloaded, offline
    return smp.Unet(
        encoder_name     = encoder_name,
        encoder_weights  = encoder_weights,
        decoder_channels = decoder_channels,
        in_channels      = in_channels,
        classes          = out_classes,
        activation       = None,
    )


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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """
        Returns (loss, logits, vessel_mask). The corresponding vessel mask is applied inside each loss to exclude background pixels.
        """

        images       = batch["patch"]                           # (B, C, H, W)
        vessel_mask  = batch["vessel_mask"].squeeze(1).long()   # (B, H, W)
        tissue_mask  = batch["tissue_mask"]                     # (B, H, W) 

        logits    = self(images)              # (B, 1, H, W)  raw logits
        logits_sq = logits.squeeze(1)         # (B, H, W)

        prob = torch.sigmoid(logits_sq)       # (B, H, W)     [0, 1]

        loss = (
            float(self.hparams['ftl_weight']) * 
            self.ftl(prob, vessel_mask, tissue_mask)
            + (1 - float(self.hparams['ftl_weight'])) * 
            self.dice_loss(prob, vessel_mask, tissue_mask)
        )

        return loss, logits_sq, vessel_mask
    

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch)

        self.log('train/loss', loss, on_step = True, on_epoch = True,
                 prog_bar = True, sync_dist = False)

        return loss
    
    
    def on_train_epoch_end(self) -> None:
        """Add an indicator the epoch ended without the messy progress bar."""
        epoch = self.current_epoch
        loss = self.trainer.callback_metrics.get('train/loss_epoch', 'N/A')
        print(f"Epoch {epoch} complete | train/loss: {loss:.4f}", flush=True)
        return

    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits_sq, vessel_mask = self._shared_step(batch)

        # Hard predictions for metric computation
        preds = (torch.sigmoid(logits_sq) > 0.5).long()     # (B, H, W)  {0, 1}

        self.val_dice.update(preds, vessel_mask)
        self.val_iou.update(preds, vessel_mask)

        self.log('val/loss', loss, on_step = False, on_epoch = True,
                 prog_bar = True, sync_dist = False)
        
        return
    
    def on_validation_epoch_end(self) -> None:
        self.log('val/dice', self.val_dice.compute(), prog_bar = False)
        self.log('val/iou', self.val_iou.compute(), prog_bar = False)
        self.val_dice.reset()
        self.val_iou.reset()

        metrics = self.trainer.callback_metrics
        print(
            f"Epoch {self.current_epoch} | "
            f"val/loss: {metrics.get('val/loss', 'N/A'):.4f} | "
            f"val/dice: {metrics.get('val/dice', 'N/A'):.4f} | "
            f"val/iou: {metrics.get('val/iou', 'N/A'):.4f}",
            flush=True
        )

        return


    def configure_optimizers(self) -> OptimizerLRScheduler:

        optimizer = torch.optim.AdamW(
            params       = self.model.parameters(),
            lr           = float(self.hparams['lr']),
            weight_decay = float(self.hparams['weight_decay'])
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer,
            T_max     = int(self.hparams['T_max']),
            eta_min   = float(self.hparams['lr']) * 1e-2
        )

        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "monitor":   "val/loss"
            }
        }
    
# [END]