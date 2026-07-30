# ==============================================================================
# Script:           inference.py
# Purpose:          Perform model inference on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/22/2026
# ==============================================================================

import torch
import h5py

import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from collections.abc import Sized
from typing import cast

from chip_stroma.utils.loggers import setup_logger
from chip_stroma.models.loss import dice_score

logger = setup_logger(__name__)


def infer_fold(model     : torch.nn.Module, 
               dataloader: DataLoader,
               device    : torch.device,
               out_path  : Path): 
    """
    Streams probs/gt to HDF5 incrementally — avoids holding full-cohort
    arrays in RAM. Probs quantized to uint8 (2 decimal precision loss,
    acceptable for threshold sweep; ~4x smaller than float32).
    """

    # Compute per-patch metrics for the full validation cohort
    per_patch_results = []
    n_patches         = len(cast(Sized, dataloader.dataset))
    h, w              = dataloader.dataset[0]['patch'].shape[-2:]
    logger.info(f"- Detected {len(dataloader)} batches for processing")

    # Pre-allocate on-disk arrays; nothing held in Python RAM across batches
    with h5py.File(out_path, "w") as h5f:
        probs_ds = h5f.create_dataset(
            "probs", shape = (n_patches, h, w), dtype = "uint8",
            chunks = (1, h, w), compression = "gzip", compression_opts = 4
        )
        gt_ds = h5f.create_dataset(
            "gt", shape = (n_patches, h, w), dtype = "bool",
            chunks = (1, h, w), compression = "gzip", compression_opts = 4
        )

        idx = 0
        with torch.no_grad():
            for batch in dataloader:

                # Fetch all information returned from the patch dataset
                images       = batch['patch'].to(device)
                vessel_masks = batch["vessel_mask"].squeeze(1).long().to(device)
                tissue_masks = batch["tissue_mask"].long().to(device)
                sample_ids   = batch["sample_id"]
                patch_names  = batch['patch_name']

                probs = torch.sigmoid(model(images).squeeze(1))
                preds = (probs > 0.5).long()

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

                # Quantize the probabilities from [0,1] -> [0,255]
                bsz                   = len(patch_names)
                probs_ds[idx:idx+bsz] = ((probs_m.detach().cpu().numpy() * 255)
                                         .astype(np.uint8))
                gt_ds[idx:idx+bsz]    = (vessel_masks_m.detach().cpu().numpy()
                                         .astype(bool))
                idx += bsz
                
                # Move entire batch to CPU/numpy once
                dice_np       = dice_batch.detach().cpu().numpy()
                precision_np  = precision.detach().cpu().numpy()
                recall_np     = recall.detach().cpu().numpy()
                has_signal_np = has_signal.detach().cpu().numpy()

                for i in range(bsz):
                    per_patch_results.append({
                        'sample_id' : sample_ids[i],
                        'patch_name': patch_names[i],
                        'dice'      : dice_np[i],
                        'precision' : precision_np[i],
                        'recall'    : recall_np[i],
                        'has_signal': has_signal_np[i],
                    })

                # Explicit cleanup of GPU tensors for each batch
                del images, vessel_masks, tissue_masks, probs, preds
                del probs_m, preds_m, vessel_masks_m
                torch.cuda.empty_cache()

    return pd.DataFrame(per_patch_results)


# [END]