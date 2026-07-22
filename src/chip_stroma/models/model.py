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

import time
import wandb
import torch

import segmentation_models_pytorch as smp
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.nn as nn
import numpy as np

from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from typing import cast
from collections import defaultdict
from pytorch_lightning.loggers import WandbLogger

from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.model_utils import compute_dist_map, macro
from chip_stroma.models.loss import (
    FocalTverskyLoss, 
    MaskedDiceLoss, 
    per_sample_dice,
    boundary_loss,
    per_sample_surface_dice
)

logger = setup_logger(__name__)
last_epoch_time = time.time()
start_time      = time.time()

# Toggle debugging logs for training/validation steps
LOG_TIME                = False
LOG_TRAIN_STEP          = False
LOG_TRAIN_STEP_INTERVAL = 500
LOG_DEBUGGING           = False


# =====| Model Creation Entry-Point |===========================================

def build_model(encoder_name:     str = "resnet34",
                encoder_weights:  str = "imagenet",
                decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
                in_channels:      int = 3,
                out_classes:      int = 1) -> nn.Module:
    """Factory function for smp U-Net vessel segmentation model."""

    # Return the pretrained ImageNet encoder; loads pre-downloaded, offline
    return smp.Unet(
        encoder_name     = encoder_name,
        encoder_weights  = encoder_weights,
        decoder_channels = decoder_channels,
        in_channels      = in_channels,
        classes          = out_classes,
        activation       = None
    )


# =====| Lightning Module |=====================================================

class VesselSegModule(pl.LightningModule):

    def __init__(self,
                 model           : nn.Module,
                 lr              : float = 1e-4,
                 weight_decay    : float = 1e-2,
                 T_max           : int   = 60,
                 ftl_alpha       : float = 0.3,
                 ftl_beta        : float = 0.7,
                 ftl_gamma       : float = 0.75,
                 ftl_weight      : float = 0.5,
                 smooth          : float = 1.0,
                 nsd_tolerance   : float = 2.0,
                 bl_weight_target: float = 0.1,
                 bl_ramp_epochs  : int = 10)   : 
        
        super().__init__()
        self.save_hyperparameters(ignore = ['model'])
        self.model = model

        # Loss: FTL + Dice, with a linearly-ramped boundary term
        self.ftl = FocalTverskyLoss(ftl_alpha, ftl_beta, ftl_gamma, smooth)
        self.dice_loss = MaskedDiceLoss(smooth)
        self.bl_weight = 0.0

        # Per-Sample Macro Accumulators; primary metrics
        self.val_sample_dice      = defaultdict(list)
        self.val_sample_nsd       = defaultdict(list)
        self.val_sample_iou       = defaultdict(list)
        self.val_sample_precision = defaultdict(list)
        self.val_sample_recall    = defaultdict(list)

        # Global/epoch micro-accumulators; diagnostic metrics
        self.val_precision_micro = BinaryPrecision()
        self.val_recall_micro    = BinaryRecall()
        self.val_iou_micro       = MeanIoU(
            num_classes        = 2,
            include_background = False,
            per_class          = False,
            input_format       = "index"
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # =====| Debugging Statements |=============================================

    def on_train_start(self):
        logger.info("-" * 50)
        logger.info("-- | Starting model training")

    def on_train_end(self):
        logger.info("-- | Completing model training")
        logger.info("-" * 50)


    # =====| Loss Step |========================================================
    
    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """
        Returns (loss, logits, vessel_mask). The corresponding vessel mask is applied inside each loss to exclude background pixels.
        """

        images       = batch["patch"]                           # (B, C, H, W)
        vessel_mask  = batch["vessel_mask"].squeeze(1).long()   # (B, H, W)
        tissue_mask  = batch["tissue_mask"]                     # (B, H, W) 

        logits    = self(images)
        logits_sq = logits.squeeze(1)

        # Loss is FTL + Dice composite weighted by ftl_weight
        f_w = float(self.hparams['ftl_weight'])
        loss = (
            f_w * self.ftl(logits_sq, vessel_mask, tissue_mask) +
            (1 - f_w) * self.dice_loss(logits_sq, vessel_mask, tissue_mask)
        )

        # Boundary loss aded once its ramp weight is greater than zero
        if self.bl_weight > 0:
            probs    = torch.sigmoid(logits_sq) * tissue_mask
            dist_map = compute_dist_map(vessel_mask) * tissue_mask
            loss     = loss + self.bl_weight * boundary_loss(probs, dist_map)

        return loss, logits_sq, vessel_mask

    # =====| Training Step |====================================================

    def on_train_epoch_start(self) -> None:
        """
        Linearly ramp boundary-loss weight from zero to target over 
        bl_ramp_epochs, then hold constant.
        """

        ramp_epochs = int(self.hparams['bl_ramp_epochs'])
        target      = float(self.hparams['bl_weight_target'])

        # Ramp the boundary loss weight up to the target over the ramp epochs
        self.bl_weight = target * min(self.current_epoch / ramp_epochs, 1.0)
        self.log('train/bl_weight', self.bl_weight, 
                 on_step = False, on_epoch = True)
        return


    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Log training loss and log X steps if toggled."""

        global last_log_time
        loss, _, vessel_mask = self._shared_step(batch)

        self.log('train/loss', loss, on_step = True, on_epoch = True,
                 prog_bar = False, batch_size = vessel_mask.shape[0])

        # Log every X steps
        if LOG_TRAIN_STEP:
            if batch_idx % LOG_TRAIN_STEP_INTERVAL == 0:
                logger.info(f"Epoch {self.current_epoch} | Step {batch_idx} | "
                            f"train/loss: {loss:.4f}")

        return loss
    

    def on_train_epoch_end(self) -> None:
        """Add an indicator when the epoch ends without messy progress bar."""

        global last_epoch_time
        global start_time
        now = time.time()
        
        epoch = self.current_epoch
        loss = self.trainer.callback_metrics.get('train/loss_epoch', 'N/A')

        logger.info(f"-- | Epoch {epoch} Complete   | train/loss: {loss:.4f} | "
                    f"Time elapsed: {now - last_epoch_time:.1f}s | "
                    f"Total runtime: {now - start_time:.1f}s")
        last_epoch_time = now
        return


    # =====| Validation Step |==================================================
    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits_sq, vessel_mask = self._shared_step(batch)

        # Hard predictions for metric computation
        preds       = (torch.sigmoid(logits_sq) > 0.5).long()
        sample_ids  = batch['sample_id']
        tissue_mask = batch['tissue_mask'].long()

        # Restrict preds/target to tisuse region before metric computation
        preds_m       = preds * tissue_mask
        vessel_mask_m = vessel_mask * tissue_mask

        # Guard against empty-union samples (no positive pixels in preds/target)
        has_signal = vessel_mask_m.sum(dim = (1, 2)) > 0
        if not has_signal.any():
            self.log('val/loss', loss, on_step = False, on_epoch = True,
                     batch_size = preds.shape[0])
            return

        # Extract signal and one-hot versions for metric computation
        preds_s, target_s = preds_m[has_signal], vessel_mask_m[has_signal]
        preds_oh  = F.one_hot(preds_s,  num_classes = 2).permute(0, 3, 1, 2)
        target_oh = F.one_hot(target_s, num_classes = 2).permute(0, 3, 1, 2)

        # Compute per-sample scores, one pass over teh signal subset
        tolerance   = float(self.hparams['nsd_tolerance'])
        sample_nsd  = per_sample_surface_dice(preds_oh, target_oh, tolerance)
        sample_dice = per_sample_dice(preds_s, target_s, num_classes = 2, 
                                      include_background = False)
        
        # Compute precision, recall, and IoU manually
        dims        = (1, 2)
        tp          = (preds_s & target_s.bool()).sum(dim=dims).float()
        fp          = (preds_s.bool() & ~target_s.bool()).sum(dim=dims).float()
        fn          = (~preds_s.bool() & target_s.bool()).sum(dim=dims).float()
        iou_s       = tp / (tp + fp + fn + 1e-8)
        precision_s = tp / (tp + fp + 1e-8)
        recall_s    = tp / (tp + fn + 1e-8)

        # Assign per-patient macros as the primary training metrics
        signal_ids = [s for s, k in zip(sample_ids, has_signal.tolist()) if k]
        for i, pid in enumerate(signal_ids):
            self.val_sample_dice[pid].append(sample_dice[i].item())
            self.val_sample_nsd[pid].append(sample_nsd[i].item())
            self.val_sample_iou[pid].append(iou_s[i].item())
            self.val_sample_precision[pid].append(precision_s[i].item())
            self.val_sample_recall[pid].append(recall_s[i].item())

        # Update global/epoch micro accumulation metrics for diagnostics
        self.val_iou_micro.update(preds_s, target_s)
        self.val_precision_micro.update(preds_s, target_s)
        self.val_recall_micro.update(preds_s, target_s)

        self.log('val/loss', loss, on_step = False, on_epoch = True,
                 batch_size = preds.shape[0])
        
        return


    def on_validation_epoch_end(self) -> None:

        # Macro reduction; nanmean within samples, then across samples
        val_dice     , val_dice_std = macro(self.val_sample_dice)
        val_nsd      , val_nsd_std  = macro(self.val_sample_nsd)
        val_iou      , _            = macro(self.val_sample_iou)
        val_precision, _            = macro(self.val_sample_precision)
        val_recall   , _            = macro(self.val_sample_recall)

        self.log('val/dice',      val_dice,      prog_bar = False)
        self.log('val/dice_std',  val_dice_std,  prog_bar = False)
        self.log('val/nsd',       val_nsd,       prog_bar = False)
        self.log('val/nsd_std',   val_nsd_std,   prog_bar = False)
        self.log('val/iou',       val_iou,       prog_bar = False)
        self.log('val/precision', val_precision, prog_bar = False)
        self.log('val/recall',    val_recall,    prog_bar = False)

        # Micro-diagnostics; logged separately, never used for selection
        self.log('val/iou_micro',       self.val_iou_micro.compute(), 
                                        prog_bar = False)
        self.log('val/precision_micro', self.val_precision_micro.compute(), 
                                        prog_bar = False)
        self.log('val/recall_micro',    self.val_recall_micro.compute(),    
                                        prog_bar = False)
        
        # Per-patient Dice table for post-hoc inter-patient variance inspection
        per_patient_dice = {pid: np.nanmean(v) for pid, v in 
                            self.val_sample_dice.items()}
        
        if isinstance(self.trainer.logger, WandbLogger):
            self.trainer.logger.experiment.log({
                'val/dice_per_patient': wandb.Table(
                    columns = ['sample_id', 'mean_dice', 'n_patches'],
                    data    = [[pid, m, len(self.val_sample_dice[pid])]
                                for pid, m in per_patient_dice.items()]
                )})
            
        # Clear all per-epoch state
        for d in (self.val_sample_dice, self.val_sample_nsd, 
                  self.val_sample_iou, self.val_sample_precision, 
                  self.val_sample_recall):
            d.clear()

        self.val_iou_micro.reset()
        self.val_precision_micro.reset()
        self.val_recall_micro.reset()

        metrics = self.trainer.callback_metrics
        logger.info(
            f"-- | Epoch {self.current_epoch} Validation | "
            f"val/loss: {metrics.get('val/loss', 'N/A'):.4f} | "
            f"val/dice: {metrics.get('val/dice', 'N/A'):.4f} | "
            f"val/iou: {metrics.get('val/iou', 'N/A'):.4f} | "
            f"val/precision: {metrics.get('val/precision', 'N/A'):.4f} | "
            f"val/recall: {metrics.get('val/recall', 'N/A'):.4f}"
        )

        return


    # =====| Optimizers |=======================================================

    def configure_optimizers(self) -> OptimizerLRScheduler:

        lr = float(self.hparams['lr'])
        wd = float(self.hparams['weight_decay'])

        param_groups = (
            self.get_groups(cast(nn.Module, self.model.encoder), lr * 0.1, wd) +
            self.get_groups(cast(nn.Module, self.model.decoder), lr, wd) +
            self.get_groups(cast(nn.Module, self.model.segmentation_head),lr,wd)
        )

        optimizer = torch.optim.AdamW(param_groups)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer,
            T_max     = int(self.hparams['T_max']),
            eta_min   = lr * 1e-2
        )

        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "monitor":   "val/loss"
            }
        }
    

    def get_groups(self, 
                   module: nn.Module, 
                   lr: float, 
                   weight_decay: float) -> list[dict]:
        """Splits params into weight-decay / no-decay groups (biases, norms)."""

        decay, no_decay = [], []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

# [END]