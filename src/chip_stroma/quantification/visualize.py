# ==============================================================================
# Script:           visualize.py
# Purpose:          Export masks overlays of fibroblasts and vessels
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import torch

import matplotlib.pyplot as plt

from pathlib import Path


def save_overlays(model, dataloader, device, patch_df, out_dir, n_examples=3):
    """
    Best/median/worst overlay panels (prediction vs. GT vs. raw) for pathologist review.
    """

    valid = patch_df.dropna(subset=["dice"]).sort_values("dice")
    picks = {
        "worst": valid.head(n_examples),
        "median": valid.iloc[len(valid)//2 - n_examples//2 : len(valid)//2 + n_examples//2 + 1],
        "best": valid.tail(n_examples),
    }
    overlay_dir = Path(out_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
 
    # Build a lookup for quick re-fetch of specific patches
    ds = dataloader.dataset
    for category, subset in picks.items():
        for _, row in subset.iterrows():
            item = ds.get_by_id(row["patch_id"])   # user's dataset must implement this
            img, mask = item["image"], item["mask"]
            with torch.no_grad():
                pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
 
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(img.permute(1, 2, 0).numpy()); axes[0].set_title("raw")
            axes[1].imshow(mask.squeeze().numpy(), cmap="gray"); axes[1].set_title("GT")
            axes[2].imshow(pred > 0.5, cmap="gray"); axes[2].set_title(f"pred (dice={row['dice']:.2f})")
            for ax in axes: ax.axis("off")
            plt.tight_layout()
            plt.savefig(overlay_dir / f"{category}_{row['patch_id']}.png", dpi=150)
            plt.close(fig)
 
 
def save_calibration_plot(patch_df, out_dir):
    """Histogram of predicted probabilities — catches double-sigmoid bug (mass collapsed near 0.5)."""
    plt.figure(figsize=(5, 4))
    plt.hist(patch_df["mean_prob"], bins=30)
    plt.xlabel("mean predicted probability (patch)")
    plt.ylabel("count")
    plt.title("Probability calibration sanity check")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "prob_calibration.png", dpi=150)
    plt.close()

# [END]