# ==============================================================================
# Script:           segmentation_plots.py
# Purpose:          Segmentation fidelity plots
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import logging

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

logging.getLogger("matplotlib.category").setLevel(logging.ERROR)


def plot_fold_boxplots(per_fold : pd.DataFrame, 
                       metric   : str        = "dice",
                       save_path: str | None = None) -> None:
    """
    Boxplot of per-patient scores grouped by fold.

    X-axis: fold (0-4). Y-axis: metric (e.g. dice). Each point is one
    patient's score in that fold's held-out set; each box summarizes the
    distribution of patients within a fold — not a per-patient breakdown.

    Purpose: check cross-fold consistency (CV stability) and flag any
    fold whose held-out patients perform anomalously.
    """
    
    fig, ax = plt.subplots(figsize = (4, 4))
    sns.boxplot(data = per_fold, x = "fold", y = metric, ax = ax, 
                showfliers = False, width = 0.5, color = "whitesmoke")
    sns.stripplot(data = per_fold, x = "fold", y = metric, ax = ax,
                  color = "black", size = 4, jitter = 0.1, alpha = 0.7)
    ax.set_title(f"Per-Patient {metric.capitalize()} Score by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path: fig.savefig(save_path, dpi = 300); plt.close(fig)
    return None


def plot_pr_curve(threshold_sweep: pd.DataFrame, 
                  save_path      : str | None = None):
    """
    Precision/recall/dice vs. classification threshold, pooled across
    all validation pixels (not per-patient or per-fold).

    X-axis: threshold (0.1-0.9). Y-axis: score.

    Purpose: justify a chosen operating threshold (e.g. vs. Otsu) instead
    of defaulting to 0.5.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    for col, style in [("precision", "--"), ("recall", "--"), ("dice", "-")]:
        ax.plot(threshold_sweep["threshold"], threshold_sweep[col], style, 
                label=col.capitalize())
        
    best = threshold_sweep.loc[threshold_sweep["dice"].idxmax()]
    
    ax.set_title("Core Metric Performance Across Classification Thresholds",
                pad = 25)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend(
        loc="lower center",
        bbox_to_anchor = (0.5, 0.98),
        ncol = 3,
        frameon=False
    )
    
    fig.tight_layout()

    if save_path: fig.savefig(save_path, dpi = 300); plt.close(fig)
    return None



def plot_patient_dice_violin(per_patient      : pd.DataFrame, 
                             highlight_patient: str ="h-BMO-18", 
                             save_path        : str | None = None):
    """
    Single violin of per-patient dice, pooled across all folds.

    Y-axis: dice, one point per patient. The highlighted patient
    (default h-BMO-18) is marked red.

    Purpose: show overall patient-level dice spread and check whether
    the known high-vessel-density outlier patient is also a performance
    outlier.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.violinplot(data=per_patient, y="dice", ax=ax, cut=0, 
                   linewidth=1, color="whitesmoke", width = 0.6)
    sns.stripplot(data=per_patient, y="dice", ax=ax, color="black", size=3, 
                  alpha=0.5, jitter=0.15)
    
    q1, q2, q3 = per_patient["dice"].quantile([0.25, 0.5, 0.75])
    
    for q, c in [(q1, "blue"), (q2, "red"), (q3, "green")]:
        ax.axhline(q, color=c, linewidth=1.5, linestyle=":", alpha=0.5)
    
    ax.legend(
        handles=[
            Line2D([0], [0], color="blue", lw=2, label="Q1 (25%)"),
            Line2D([0], [0], color="red", lw=2, label="Median"),
            Line2D([0], [0], color="green", lw=2, label="Q3 (75%)")
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.3
    )
    
    hl = per_patient[per_patient["sample_id"] == highlight_patient]
    if not hl.empty:
        ax.scatter([0] * len(hl), hl["dice"], color="red", zorder=5, 
                   label=highlight_patient, s=60)
        ax.legend(frameon=False)
    
    ax.set_title("Per-Patient Dice Score Across All Folds", pad=15)
    ax.set_ylabel("Dice Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    
    if save_path: fig.savefig(save_path, dpi = 300); plt.close(fig)
    return None


def plot_overlay_panel(image     : np.ndarray, 
                       gt_mask   : np.ndarray,
                       pred_mask : np.ndarray,
                       dice_score: float,
                       save_path : str | None = None):
    """
    3-panel QC image for a single patch: raw image | ground-truth
    contour | predicted contour.

    Purpose: qualitative visual inspection of segmentation quality for
    a selected best/median/worst-dice case — not a summary statistic.
    """

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image); axes[0].set_title("Patch"); axes[0].axis("off")
    
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, cmap="Reds", alpha=0.3)
    axes[1].contour(gt_mask, colors="darkred", linewidths=1)
    axes[1].set_title("Ground Truth Annotation")
    axes[1].axis("off")
    
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, cmap="Reds", alpha=0.3)
    axes[2].contour(pred_mask, colors="darkred", linewidths=1)
    axes[2].set_title(f"Vessel Prediction (Dice={dice_score:.3f})")
    axes[2].axis("off")
    
    fig.suptitle("Predicted Segmentation Compared with Ground Truth Mask", 
                 fontsize = 16, y=1.02)
    fig.tight_layout()

    if save_path: fig.savefig(save_path, dpi = 300); plt.close(fig)
    return None

def plot_optuna_importance(importance: pd.DataFrame, 
                           save_path : str | None = None):
    """
    Horizontal bar chart of hyperparameter importance.

    Y-axis: hyperparameter name. X-axis: fANOVA importance score (0-1),
    sorted descending.

    Purpose: identify which hyperparameters most influenced val/dice
    across the HPO sweep.
    """
    names = {
        "lr": "Learning Rate",
        "bl_weight_target": "Boundary Loss Weight",
        "gradient_clip_val": "Gradient Clipping",
        "ftl_weight": "Focal Tversky Loss Weight",
        "bl_ramp_epochs": "Boundary Loss Ramp Epochs",
        "ftl_alpha": "Focal Tversky Loss Alpha"
    }
    
    importance["param"] = importance["param"].replace(names)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    importance = importance.sort_values("importance")
    ax.barh(importance["param"], importance["importance"], color="lightblue", 
            height = 0.5)
    ax.set_xlabel("fANOVA Importance")
    ax.set_title("Hyperparameter Importance", fontsize = 14, pad = 12.5)
    fig.tight_layout()

    if save_path: fig.savefig(save_path, dpi = 300); plt.close(fig)
    return None

def plot_confusion_matrix(threshold_sweep_counts: dict, save_path: str | None = None):
    """
    Normalized 2x2 pixel-level confusion matrix at the operating threshold
    (argmax-dice from threshold_sweep).

    Cells: TP/FP/FN/TN as fraction of all tissue-masked pixels.

    Purpose: more granular than dice alone — shows whether errors skew
    false-positive or false-negative, informing whether the vessel
    density score (script 10) will over- or under-estimate area.
    """
    tp, fp, fn, tn = threshold_sweep_counts["tp"], threshold_sweep_counts["fp"], threshold_sweep_counts["fn"], threshold_sweep_counts["tn"]
    total = tp + fp + fn + tn
    matrix = np.array([[tn, fp], [fn, tp]]) / total

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=["Pred: bg", "Pred: vessel"], yticklabels=["GT: bg", "GT: vessel"], ax=ax)
    ax.set_title("Pixel-level confusion matrix (normalized)")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300); plt.close(fig)
    return fig


def plot_calibration(probs: np.ndarray, 
                     gt: np.ndarray, 
                     n_bins: int = 10, 
                     save_path: str | None = None):
    """
    Reliability diagram: predicted probability vs. observed positive
    frequency, binned into n_bins.

    X-axis: mean predicted probability per bin. Y-axis: observed
    fraction of true-positive pixels in that bin. Diagonal = perfect
    calibration.

    Purpose: Otsu thresholding for vessel density scoring assumes
    probabilities are meaningfully ordered/scaled — miscalibration here
    would bias downstream density scores and the CHIP vs. non-CHIP
    t-test.
    """
    probs_flat, gt_flat = probs.ravel(), gt.ravel().astype(bool)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs_flat, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_pred, obs_freq = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            mean_pred.append(probs_flat[mask].mean())
            obs_freq.append(gt_flat[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, obs_freq, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300); plt.close(fig)
    return fig


def plot_dice_vs_vessel_area(per_patient: pd.DataFrame, 
                             save_path: str | None = None):
    """
    Scatter of per-patient dice vs. ground-truth vessel-area fraction.

    X-axis: fraction of tissue pixels that are vessel-positive (GT).
    Y-axis: per-patient mean dice.

    Purpose: checks whether performance degrades on low-vessel-density
    patients — the primary failure mode expected under ~125:1 class
    imbalance, and directly relevant since vessel area is your
    scientific endpoint (CHIP vs. non-CHIP comparison).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(per_patient["vessel_area_frac"], per_patient["dice"], color="steelblue", alpha=0.7)
    for _, row in per_patient.iterrows():
        ax.annotate(row["sample_id"], (row["vessel_area_frac"], row["dice"]), fontsize=6, alpha=0.6)
    ax.set_xlabel("GT vessel-area fraction"); ax.set_ylabel("Mean dice")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300); plt.close(fig)
    return fig
    
# [END]