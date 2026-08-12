# ==============================================================================
# Script:           quantification_plots.py
# Purpose:          General analysis plots
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/12/2026
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import gaussian_kde


def plot_chip_boxplot(sample_density: pd.DataFrame,
                      col           : str,
                      welsh_p,
                      mwu_p         : float,
                      eff_g         : float,
                      out_path      : Path) -> None:
    
    fig, ax = plt.subplots(figsize = (5, 4))
    
    sns.boxplot(data = sample_density, x = "chip_status", y = col, 
                ax = ax, showfliers = False)
    sns.swarmplot(data = sample_density, x = "chip_status", y = col,
                  ax = ax, color = "k", size = 4)
    
    ax.set_xticklabels(["non-CHIP", "CHIP"])
    ax.set_title(f"Welch p = {welsh_p:.3f}, "
                 f"MWU p = {mwu_p:.3f}, "
                 f"g = {eff_g:.2f}")

    fig.tight_layout()
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)


def plot_loocv_roc_curve(loocv: dict, out_path: Path) -> None:

    fig, ax = plt.subplots(figsize = (5, 4))

    ax.plot(loocv["fpr"], loocv["tpr"], 
            label = f"AUC = {loocv['auc']:.2f}\nperm p = {loocv['perm_p']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color = "gray")

    ax.set_xlabel("FPR") 
    ax.set_ylabel("TPR")
    ax.set_title("LOOCV ROC")
    
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)


def plot_kde_overlap(sample_density: pd.DataFrame,
                     col           : str,
                     chip          : pd.DataFrame,
                     non_chip      : pd.DataFrame,
                     out_path      : Path) -> None: 
    
    xs = np.linspace(sample_density[col].min(), sample_density[col].max(), 200)
    kde_g  = gaussian_kde(chip)(xs)
    kde_ng = gaussian_kde(non_chip)(xs)

    fig, ax = plt.subplots(figsize = (5, 4))

    ax.plot(xs, kde_g, label = "CHIP", color = "firebrick")
    ax.plot(xs, kde_ng, label = "non-CHIP", color = "steelblue")
    ax.fill_between(xs, np.minimum(kde_g, kde_ng), alpha = 0.3, 
                    color = "gray", label = "overlap")
    ax.set_xlabel(col)
    ax.set_title("Distribution Overlap")

    ax.legend(fontsize = 8)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)


def plot_loocv_distribution(loocv: dict, out_path: Path) -> None:
    
    probs  = np.asarray(loocv["probs"])
    y_true = np.asarray(loocv["y_true"])

    fig, ax = plt.subplots(figsize=(5, 2.5))
    for lbl, color in [(1, "firebrick"), (0, "steelblue")]:
        sns.stripplot(x = probs[y_true == lbl], y = [0] * (y_true == lbl).sum(),
                      ax = ax, jitter = 0.15, size = 6, color = color,
                      label = "CHIP" if lbl == 1 else "non-CHIP")
        
    ax.axvline(loocv["youden_cutoff"], linestyle = "--", color = "k", 
               label = "Youden cutoff")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("LOOCV predicted P(CHIP)")
    ax.set_title(f"Predicted probabilities (sens = {loocv['sens']:.2f}, "
                 f"spec = {loocv['spec']:.2f})")
    ax.legend(fontsize = 8, loc = "upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)

# [END]