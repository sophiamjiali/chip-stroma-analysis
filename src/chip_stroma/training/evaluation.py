# ==============================================================================
# Script:           evaluation.py
# Purpose:          Evaluation utilities for UNet model training
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/09/2026
# ==============================================================================

import torch
import pandas as pd
import numpy as np


def compute_patch_metrics(model, dataloader, device, threshold = 0.5):
    """
    Runs inference; compute per-patch Dice/precision/recall with NaN-exclusion 
    for empty-mask patches.
    """

    dice_metric = DiceMetric(reduction = "none")
    f1_metric = BinaryF1Score().to(device)

    rows = []
    model.eval()

    # Compute metrics for all patches in the dataloader
    with torch.no_grad():
        for batch in dataloader:

            # Extract all relevant information from the batch
            x, y, patient_ids, patch_ids = (
                batch["image"].to(device),
                batch["mask"].to(device),
                batch["patient_id"],
                batch["patch_id"],
            )

            # Perform inference on the batch using the model 
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
 
            for i in range(x.shape[0]):

                # Do not include patches without signal
                has_signal = y[i].sum() > 0
                if not has_signal: dice = float("nan")
                else: dice = dice_metric(preds[i:i+1], y[i:i+1]).item()
 
                # Compute precision and recall on the inference
                tp = ((preds[i] == 1) & (y[i] == 1)).sum().item()
                fp = ((preds[i] == 1) & (y[i] == 0)).sum().item()
                fn = ((preds[i] == 0) & (y[i] == 1)).sum().item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
 
                rows.append({
                    "patient_id": patient_ids[i],
                    "patch_id"  : patch_ids[i],
                    "dice"      : dice,
                    "precision" : precision,
                    "recall"    : recall,
                    "has_signal": has_signal.item(),
                    "mean_prob" : probs[i].mean().item(),
                })

    return pd.DataFrame(rows)

def threshold_sweep(model, 
                    dataloader, 
                    device, 
                    thresholds = np.arange(0.1, 1.0, 0.1)) -> pd.DataFrame:
    """
    Dice vs. threshold curve — informs downstream Otsu/fixed-threshold choice.
    """
    results = []
    for t in thresholds:
        df = compute_patch_metrics(model, dataloader, device, threshold=t)
        results.append({
            "threshold": t, 
            "mean_dice": df["dice"].mean(skipna=True)
        })
    return pd.DataFrame(results)


# [END]