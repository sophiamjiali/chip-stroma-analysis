# ==============================================================================
# Script:           segmentation_plots.py
# Purpose:          Segmentation fidelity plots
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/23/2026
# ==============================================================================

import pandas as pd
import numpy as np

def plot_fold_boxplots(per_fold: pd.DataFrame, 
                       metric: str = "dice", 
                       save_path: str | None = None):
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


def plot_boundary_loss_ablation(with_bl_curve: pd.DataFrame, 
                                without_bl_curve: pd.DataFrame, 
                                ramp_epoch: pd.DataFrame, 
                                save_path : str | None = None):
    return

# [END]