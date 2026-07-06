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
import torch

import segmentation_models_pytorch as smp
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.nn as nn
import numpy as np

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

from torchmetrics.segmentation import MeanIoU, DiceScore
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from typing import cast
from collections import defaultdict

from chip_stroma.utils.loggers import setup_logger
from chip_stroma.models.loss import FocalTverskyLoss, MaskedDiceLoss

logger = setup_logger(__name__)
last_log_time   = time.time()
last_epoch_time = time.time()
start_time      = time.time()

# Toggle debugging logs for training/validation steps
LOG_TIME                = False
LOG_TRAIN_STEP          = False
LOG_TRAIN_STEP_INTERVAL = 500


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
                 model:         nn.Module,
                 lr:            float = 1e-4,
                 weight_decay:  float = 1e-2,
                 T_max:         int   = 25,
                 ftl_alpha:     float = 0.3,
                 ftl_beta:      float = 0.7,
                 ftl_gamma:     float = 0.75,
                 ftl_weight:    float = 0.5,
                 smooth:        float = 1.0,
                 nsd_tolerance: float = 2.0):
        
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
            aggregation_level  = "samplewise",
            input_format       = "index"
        )
        self._val_sample_dice = defaultdict(list)

        # Initialize Mean IoU as one-hot input, converted in validation step
        self.val_iou = MeanIoU(
            num_classes        = 2,
            include_background = False,
            per_class          = False,
            input_format       = "index"
        )

        self.val_nsd = SurfaceDiceMetric(
            class_thresholds   = [nsd_tolerance],
            include_background = False
        )
        self._val_sample_nsd = defaultdict(list)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # =====| Debugging Statements |=============================================

    def on_train_start(self):
        logger.info("-" * 50)
        logger.info("-- | Starting model training")

    def on_train_end(self):
        logger.info("-- | Completing model training")
        logger.info("-" * 50)


    # =====| Steps |============================================================
    
    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """
        Returns (loss, logits, vessel_mask). The corresponding vessel mask is applied inside each loss to exclude background pixels.
        """

        images       = batch["patch"]                           # (B, C, H, W)
        vessel_mask  = batch["vessel_mask"].squeeze(1).long()   # (B, H, W)
        tissue_mask  = batch["tissue_mask"]                     # (B, H, W) 

        logits    = self(images)              # (B, 1, H, W)  raw logits
        logits_sq = logits.squeeze(1)         # (B, H, W)

        loss = (
            float(self.hparams['ftl_weight']) * 
            self.ftl(logits_sq, vessel_mask, tissue_mask)
            + (1 - float(self.hparams['ftl_weight'])) * 
            self.dice_loss(logits_sq, vessel_mask, tissue_mask)
        )

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
    
    
    def on_train_epoch_start(self):
        """Add an indicator when the epoch starts without messy progress bar."""
        global start_time

        epoch = self.current_epoch
        logger.info(f"-- | Epoch {epoch} started | "
                    f"Total runtime: {time.time() - start_time:.1f}s")
        return


    def on_train_epoch_end(self) -> None:
        """Add an indicator when the epoch ends without messy progress bar."""
        global last_epoch_time
        now = time.time()
        
        epoch = self.current_epoch
        loss = self.trainer.callback_metrics.get('train/loss_epoch', 'N/A')

        logger.info(f"-- | Epoch {epoch} complete | train/loss: {loss:.4f} | "
                    f"Time elapsed: {now - last_epoch_time:.1f}s")
        last_epoch_time = now
        return

    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits_sq, vessel_mask = self._shared_step(batch)

        # Hard predictions for metric computation
        preds = (torch.sigmoid(logits_sq) > 0.5).long()     # (B, H, W)  {0, 1}
        sample_ids = batch['sample_id']

        preds_oh  = F.one_hot(preds, num_classes = 2).permute(0, 3, 1, 2)
        target_oh = F.one_hot(vessel_mask, num_classes = 2).permute(0, 3, 1, 2)

        # Guard against empty-union samples (no positive pixels in preds/target)
        has_signal     = (preds.sum(dim = (1, 2)) + 
                          vessel_mask.sum(dim = (1, 2,))) > 0
        nsd_per_sample = torch.full(
            (preds.shape[0],), float('nan'), 
            device = preds.device,
            dtype  = preds_oh.dtype if False else torch.float32
        )
        
        if has_signal.any():
            nsd_out = self.val_nsd(preds_oh[has_signal], target_oh[has_signal])
            vals = (nsd_out if isinstance(nsd_out, torch.Tensor) 
                    else torch.stack(list(nsd_out)))
            nsd_per_sample[has_signal] = vals.squeeze(-1).float()

        per_sample_dice = self._per_sample_dice(preds, vessel_mask, 
                                                num_classes = 2, 
                                                include_background = False)
        self.val_dice.update(preds, vessel_mask)

        for i, pid in enumerate(sample_ids):
            self._val_sample_nsd[pid].append(nsd_per_sample[i].item())
            self._val_sample_dice[pid].append(per_sample_dice[i].item())

        # Only update IOU if there is at least one positive pixel in target/pred
        if (vessel_mask.sum() > 0) or (preds.sum() > 0):
            self.val_iou.update(preds, vessel_mask)

        self.log('val/loss', loss, on_step = False, on_epoch = True,
                 prog_bar = False, sync_dist = False, 
                 batch_size = preds.shape[0])
        
        return
    

    def _per_sample_dice(self, 
                         preds, 
                         target, 
                         num_classes        = 2, 
                         include_background = False, 
                         eps                = 1e-6):
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
    

    def on_validation_epoch_end(self) -> None:
        
        # Block against NaN warnings; valid outcome, not error
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = RuntimeWarning)

            dice_means = [np.nanmean(v) for v in self._val_sample_dice.values()]
            nsd_means  = [np.nanmean(v) for v in self._val_sample_nsd.values()]

            val_dice     = torch.tensor(np.nanmean(dice_means))
            val_dice_std = torch.tensor(np.nanstd(dice_means))

            val_nsd     = torch.tensor(np.nanmean(nsd_means))
            val_nsd_std = torch.tensor(np.nanstd(nsd_means))

        val_iou = (self.val_iou.compute() if self.val_iou.update_count > 0 
                   else torch.tensor(float('nan')))

        self.log('val/dice',     val_dice, prog_bar = False)
        self.log('val/dice_std', val_dice_std, prog_bar = False)
        self.log('val/nsd',      val_nsd, prog_bar = False)
        self.log('val/nsd_std',  val_nsd_std, prog_bar = False)
        self.log('val/iou',      val_iou, prog_bar = False)

        self._val_sample_dice.clear()
        self._val_sample_nsd.clear()
        self.val_iou.reset()

        metrics = self.trainer.callback_metrics
        logger.info(
            f"Epoch {self.current_epoch} | "
            f"val/loss: {metrics.get('val/loss', 'N/A'):.4f} | "
            f"val/dice: {metrics.get('val/dice', 'N/A'):.4f} | "
            f"val/dice_std: {metrics.get('val/dice_std', 'N/A'):.4f} | "
            f"val/iou: {metrics.get('val/iou', 'N/A'):.4f}"
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
    
class SurfaceDiceMetric:
    """Stateful NSD accumulator, API-compatible with MONAI's CumulativeIterationMetric."""
    def __init__(self, class_thresholds, include_background=False, spacing_mm=(1.0, 1.0)):
        self.class_thresholds = class_thresholds
        self.include_background = include_background
        self.spacing_mm = spacing_mm
        self._buffer = []

    def __call__(self, y_pred, y):
        start_c = 0 if self.include_background else 1
        batch_vals = []
        for b in range(y_pred.shape[0]):
            per_class = []
            for c in range(start_c, y_pred.shape[1]):
                sd = compute_surface_distances(
                    y[b, c].cpu().numpy().astype(bool),
                    y_pred[b, c].cpu().numpy().astype(bool),
                    self.spacing_mm,
                )
                tol = self.class_thresholds[c - start_c]
                per_class.append(compute_surface_dice_at_tolerance(sd, tol))
            batch_vals.append(per_class)
        self._buffer.extend(batch_vals)
        return torch.tensor(batch_vals)

    def aggregate(self):
        import numpy as np
        return torch.tensor(np.nanmean(self._buffer))

    def reset(self):
        self._buffer = []

# [END]