# ==============================================================================
# Script:           inference.py
# Purpose:          Perform model inference on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/22/2026
# ==============================================================================

import torch

import pandas as pd

from torch.utils.data import DataLoader

from chip_stroma.utils.loggers import setup_logger
from chip_stroma.models.loss import dice_score

logger = setup_logger(__name__)


def infer_fold(model: torch.nn.Module, 
               dataloader: DataLoader, 
               device: torch.device): 
    """Run inference for a single fold. Returns per-patch metrics and raw 
    probabilities for threshold sweep."""

    # Compute per-patch metrics for the full validation cohort
    per_patch_results, probs_store, gt_store = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            logger.info("Processing a batch...")
            
            # Fetch all information returned from the patch dataset
            images       = batch['patch'].to(device)
            vessel_masks = batch["vessel_mask"].squeeze(1).long().to(device)
            tissue_masks = batch["tissue_mask"].long().to(device)
            sample_ids   = batch["sample_id"]
            patch_names  = batch['patch_name']

            logits_sq = model(images).squeeze(1)
            probs     = torch.sigmoid(logits_sq)
            preds     = (probs > 0.5).long()

            # Restrict to tissue regions only
            probs_m        = probs * tissue_masks
            preds_m        = preds * tissue_masks
            vessel_masks_m = vessel_masks * tissue_masks

            # Exclude patches without positive vessel annotations
            has_signal = vessel_masks_m.sum(dim = (1, 2)) > 0

            # Compute per-sample Dice guarded against NaN
            dice_batch = dice_score(preds_m.float(), vessel_masks_m.float())

            # Compute precision/recall guarded against NaN
            tp       = (preds_m * vessel_masks_m).flatten(1).sum(dim = 1)
            pred_pos = preds_m.flatten(1).sum(dim = 1)
            true_pos = vessel_masks_m.flatten(1).sum(dim = 1)
            
            precision = torch.where(pred_pos > 0, tp / pred_pos, 
                                    torch.tensor(float('nan')))
            recall    = torch.where(true_pos > 0, tp / true_pos, 
                                    torch.tensor(float('nan')))
            
            # Mask for patches with positive vessel annotations
            precision = torch.where(has_signal, precision, 
                                    torch.tensor(float('nan')))
            recall    = torch.where(has_signal, recall, 
                                    torch.tensor(float('nan')))
            
            # Move entire batch to CPU/numpy once
            dice_np       = dice_batch.detach().cpu().numpy()
            precision_np  = precision.detach().cpu().numpy()
            recall_np     = recall.detach().cpu().numpy()
            has_signal_np = has_signal.detach().cpu().numpy()

            for i in range(len(patch_names)):
                per_patch_results.append({
                    'sample_id' : sample_ids[i],
                    'patch_name': patch_names[i],
                    'dice'      : dice_np[i],
                    'precision' : precision_np[i],
                    'recall'    : recall_np[i],
                    'has_signal': has_signal_np[i],
                })

            # Retain raw probabilities/GTs for threshold_sweep() in evaluation
            probs_store.append(probs_m.detach().cpu().numpy())
            gt_store.append(vessel_masks_m.detach().cpu().numpy())

    return pd.DataFrame(per_patch_results), probs_store, gt_store


# [END]