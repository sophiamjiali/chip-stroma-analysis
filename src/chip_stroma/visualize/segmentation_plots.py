# ==============================================================================
# Script:           segmentation_plots.py
# Purpose:          Segmentation fidelity plots
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def plot_fold_boxplots(per_fold : pd.DataFrame, 
                       metric   : str        = "dice",
                       save_path: str | None = None) -> None:
    """Per-patient metric distribution per fold; boxplot + overlaid points."""

    fig, ax = plt.subplots(figsize = (6, 4))
    sns.boxplot(data = per_fold, x = "fold", y = metric, ax = ax, 
                showfliers = False, color = "lightgray")
    sns.stripplot(data = per_fold, x = "fold", y = metric, ax = ax,
                  color = "black", size = 4, jitter = 0.15, alpha = 0.7)
    ax.set_title(f"Per-patient {metric} by fold")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path: fig.savefig(save_path, dpi = 300); plt.close(fig)
    return


def plot_training_curves(run_history: pd.DataFrame,
                         metrics    : tuple = ("val/loss","val/dice"), save_path  : str | None = None):
    return


def plot_pr_curve(threshold_sweep: pd.DataFrame, 
                  save_path      : str | None = None):
    return



def plot_patient_dice_violin(per_patient      : pd.DataFrame, 
                             highlight_patient: str ="h-BMO-18", 
                             save_path        : str | None = None):
    return


def plot_overlay_panel(image     : np.ndarray, 
                       gt_mask   : np.ndarray,
                       pred_mask : np.ndarray,
                       dice_score: float,
                       save_path : str | None = None): 
    return

def plot_optuna_importance(importance: pd.DataFrame, 
                           save_path : str | None =None):
    return


def plot_optuna_parallel_coords(study     : pd.DataFrame, 
                                save_path : str | None = None):
    return

# [END]