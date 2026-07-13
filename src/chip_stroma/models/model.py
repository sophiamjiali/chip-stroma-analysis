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

import warnings
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
from chip_stroma.utils.model_utils import compute_dist_map
from chip_stroma.models.loss import (
    FocalTverskyLoss, 
    MaskedDiceLoss, 
    SurfaceDiceMetric,
    per_sample_dice,
    boundary_loss
)

logger = setup_logger(__name__)
last_log_time   = time.time()
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
    """
    Factory function for smp U-Net vessel segmentation model.

    ResNet34 encoder selected for optimal capacity/overfitting tradeoff
    at n=24 samples (Raghu et al., NeurIPS, 2019; Talo et al.,
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

        # ~~~ Loss ~~~

        # Initialize Focal Tversky Loss function
        self.ftl = FocalTverskyLoss(
            alpha  = ftl_alpha,
            beta   = ftl_beta,
            gamma  = ftl_gamma,
            smooth = smooth
        )

        # Initialize masked Dice Loss function
        self.dice_loss = MaskedDiceLoss(smooth = smooth)
        self.val_sample_dice = defaultdict(list)

        # Initialize boundary loss weight as ramping
        self.bl_weight = 0.0

        # ~~~ Training Metrics ~~~

        # Initialize Mean IoU as one-hot input, converted in validation step
        self.val_iou = MeanIoU(
            num_classes        = 2,
            include_background = False,
            per_class          = False,
            input_format       = "index"
        )

        # Initialize NSD as SurfaceDiceMetric
        self.val_nsd = SurfaceDiceMetric(
            class_thresholds   = [nsd_tolerance],
            include_background = False
        )
        self.val_sample_nsd = defaultdict(list)

        # Precision and recall for positive vessels
        self.val_precision = BinaryPrecision()
        self.val_recall    = BinaryRecall()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # =====| Debugging Statements |=============================================

    def on_train_start(self):
        logger.info("-" * 50)
        logger.info("-- | Starting model training")

    def on_train_end(self):
        logger.info("-- | Completing model training")
        logger.info("-" * 50)


    # =====| Training Step |====================================================
    
    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """
        Returns (loss, logits, vessel_mask). The corresponding vessel mask is applied inside each loss to exclude background pixels.
        """

        images       = batch["patch"]                           # (B, C, H, W)
        vessel_mask  = batch["vessel_mask"].squeeze(1).long()   # (B, H, W)
        tissue_mask  = batch["tissue_mask"]                     # (B, H, W) 

        logits    = self(images)              # (B, 1, H, W)  raw logits
        logits_sq = logits.squeeze(1)         # (B, H, W)

        # Loss includes Focal Tversky, Dice Loss, and boundary loss terms
        f_w = float(self.hparams['ftl_weight'])
        loss = (
            f_w * self.ftl(logits_sq, vessel_mask, tissue_mask) +
            (1 - f_w) * self.dice_loss(logits_sq, vessel_mask, tissue_mask)
        )

        # Compute distance map masked by tissue content for boundary loss
        if self.bl_weight > 0:
            probs = torch.sigmoid(logits_sq) * tissue_mask
            dist_map = compute_dist_map(vessel_mask) * tissue_mask
            loss = loss + self.bl_weight * boundary_loss(probs, dist_map)

        return loss, logits_sq, vessel_mask

    

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        global last_log_time
        loss, _, _ = self._shared_step(batch)

        # Log every X steps
        if LOG_TRAIN_STEP:
            if batch_idx % LOG_TRAIN_STEP_INTERVAL == 0:
                logger.info(f"Epoch {self.current_epoch} | Step {batch_idx} | "
                            f"train/loss: {loss:.4f}")
            
        # Detect if the model training is stalling; only if logging is enabled
        if LOG_TIME:
            if batch_idx % 50 == 0:
                now = time.time()
                logger.info(f"Alive check | Epoch {self.current_epoch} | "
                            f"Step {batch_idx} | "
                            f"Time since last log: {now - last_log_time:.1f}s")
                last_log_time = now

        self.log('train/loss', loss, on_step = True, on_epoch = True,
                 prog_bar = False, sync_dist = False)

        return loss
    

    def on_train_epoch_start(self) -> None:
        """
        Linearly ramp boundary-loss weight from 0 to target over ramp_epochs.
        """

        ramp_epochs = int(self.hparams['bl_ramp_epochs'])
        target      = float(self.hparams['bl_weight_target'])

        # Ramp the boundary loss weight up to the target over the ramp epochs
        self.bl_weight = target * min(self.current_epoch / ramp_epochs, 1.0)
        self.log('train/bl_weight', self.bl_weight, on_step = False, 
                 on_epoch = True)
        
        return
    

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

        preds_oh  = F.one_hot(preds_m, num_classes = 2).permute(0, 3, 1, 2)
        target_oh = F.one_hot(vessel_mask_m, num_classes=2).permute(0, 3, 1, 2)

        # Guard against empty-union samples (no positive pixels in preds/target)
        has_signal = vessel_mask_m.sum(dim = (1, 2)) > 0

        # Compute NSD per sample, nan-filled for no-signal samples
        nsd_per_sample = torch.full(
            (preds.shape[0],), float('nan'),
            device = preds.device,
            dtype  = torch.float32
        )
        
        # Only update the global metric if there was signal in the sample
        if has_signal.any(): 
            nsd_out = self.val_nsd(preds_oh[has_signal], target_oh[has_signal])
            vals    = (nsd_out if isinstance(nsd_out, torch.Tensor) 
                       else torch.stack(list(nsd_out)))
            nsd_per_sample[has_signal] = (vals.squeeze(-1).float()
                                          .to(preds.device))

        # Compute per-sample dice; already has NaN-guards internally
        per_sample_dice_metric = per_sample_dice(
            preds              = preds_m,
            target             = vessel_mask_m,
            num_classes        = 2,
            include_background = False
        )

        # Aggregate IoU guarded against NaN
        if (vessel_mask.sum() > 0) or (preds.sum() > 0):
            self.val_iou.update(preds_m[has_signal], vessel_mask_m[has_signal])

        # Compute precision and recall, guarded against NaN
        if has_signal.any():
            self.val_precision.update(preds_m[has_signal], 
                                      vessel_mask_m[has_signal])
            self.val_recall.update(preds_m[has_signal], 
                                   vessel_mask_m[has_signal])

        # Track positive-patch count for imbalance sanity-checking
        self.log('val/n_pos_in_batch', has_signal.sum(), on_step = False,
                 on_epoch = True, reduce_fx = 'sum', prog_bar = False,
                 batch_size = preds.shape[0])g

        # Per-patient accumulation using nan-safe values
        for i, pid in enumerate(sample_ids):
            self.val_sample_nsd[pid].append(nsd_per_sample[i].item())
            self.val_sample_dice[pid].append(per_sample_dice_metric[i].item())

        self.log('val/loss', loss, on_step = False, on_epoch = True,
                 prog_bar = False, sync_dist = False, 
                 batch_size = preds.shape[0])
        
        return


    def on_validation_epoch_end(self) -> None:
        
        # Block against NaN warnings; valid outcome, not error
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = RuntimeWarning)

            dice_means = [np.nanmean(v) for v in self.val_sample_dice.values()]
            nsd_means  = [np.nanmean(v) for v in self.val_sample_nsd.values()]

            val_dice     = torch.tensor(np.nanmean(dice_means))
            val_dice_std = torch.tensor(np.nanstd(dice_means))

            val_nsd     = torch.tensor(np.nanmean(nsd_means))
            val_nsd_std = torch.tensor(np.nanstd(nsd_means))
        
        # Compute metrics with a signal guard
        val_iou       = (self.val_iou.compute() if self.val_iou.update_count >0 
                         else torch.tensor(float('nan')))
        val_precision = (self.val_precision.compute() if self.val_precision.
                         update_count > 0 else torch.tensor(float('nan')))
        val_recall    = (self.val_recall.compute() if self.val_recall.
                         update_count > 0 else torch.tensor(float('nan')))

        self.log('val/dice',      val_dice,      prog_bar = False)
        self.log('val/dice_std',  val_dice_std,  prog_bar = False)
        self.log('val/nsd',       val_nsd,       prog_bar = False)
        self.log('val/nsd_std',   val_nsd_std,   prog_bar = False)
        self.log('val/iou',       val_iou,       prog_bar = False)
        self.log('val/precision', val_precision, prog_bar = False)
        self.log('val/recall',    val_recall,    prog_bar = False)

        # Compute per-patient Dice table for post-hoc inter-patient variance
        per_patient_dice = {pid: np.nanmean(v) for pid, v in 
                            self.val_sample_dice.items()}
        
        if isinstance(self.trainer.logger, WandbLogger):
            self.trainer.logger.experiment.log({
                'val/dice_per_patient': wandb.Table(
                    columns = ['sample_id', 'mean_dice', 'n_patches'],
                    data = [[pid, m, len(self.val_sample_dice[pid])] for 
                            pid, m in per_patient_dice.items()]
            )})
        
        # Clear all validation metrics for the next iteration
        self.val_sample_dice.clear()
        self.val_sample_nsd.clear()
        self.val_iou.reset()
        self.val_precision.reset()
        self.val_recall.reset()

        metrics = self.trainer.callback_metrics
        logger.info(
            f"-- | Epoch {self.current_epoch} Validation | "
            f"val/loss: {metrics.get('val/loss', 'N/A'):.4f} | "
            f"val/dice: {metrics.get('val/dice', 'N/A'):.4f} | "
            f"val/dice_std: {metrics.get('val/dice_std', 'N/A'):.4f} | "
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
        decay, no_decay = [], []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    

    

# [END]