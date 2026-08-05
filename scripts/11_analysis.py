# ==============================================================================
# Script:           11_analysis.yaml
# Purpose:          Quantify fibroblasts in aSMA stains
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             08/05/2026
# ==============================================================================

import json

import argparse as ap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.classification.statistics import hedges_g, bootstrap_ci

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Analysis",
        config_path    = Path(args.config_dir) / "11_analysis.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "11_analysis.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # Load the validation fold as a dataset
    manifest = initialize_train_manifest(
        train_path = config.paths.metadata.train_manifest,
        patch_path = config.paths.metadata.patch_manifest
    )
    sample_labels = manifest[['sample_id', 'chip_status']].drop_duplicates()

    # Initialize version results directory
    analysis_dir = Path(config.paths.results) / args.version / "analysis"
    quantify_dir = Path(config.paths.results) / args.version / "quantify"
    analysis_dir.mkdir(parents = True, exist_ok = True)

    # Load and merge CHIP status into sample-level fibroblast density
    sample_density = pd.read_csv(quantify_dir /"sample_fibroblast_density.csv") 
    sample_density = sample_density.merge(
        sample_labels, 
        on       = 'sample_id',
        how      = 'left',
        validate = 'many_to_one'
    )

    # Restrict to single base threshold (exclude swept thresholds)
    sample_density = sample_density[
        sample_density['vessel_threshold'] == config.analysis.vessel_threshold
    ]

    chip     = sample_density.loc[sample_density['chip_status'] == 1, 
                                  "sample_fibroblast_density"]
    non_chip = sample_density.loc[sample_density['chip_status'] == 0, 
                                  "sample_fibroblast_density"]

    # Assumption check
    shapiro_chip = stats.shapiro(chip)[1], 
    shapiro_non  = stats.shapiro(non_chip)[1]
    levene_p     = stats.levene(chip, non_chip)[1]

    # Primary significance test
    t_stat, t_p = stats.ttest_ind(chip, non_chip, equal_var = False)

    # Non-parametric backup to T-Test
    u_stat, u_p = stats.mannwhitneyu(chip, non_chip, alternative = "two-sided")

    # Effect size
    g = hedges_g(chip, non_chip)

    # Bootstrap CI on mean difference
    ci_low, ci_high = bootstrap_ci(chip.values, non_chip.values)

    # Separation via LOOCV logistic regression and AUC
    X = sample_density[['sample_fibroblast_density']].values
    y = sample_density['chip_status'].values

    loo_probs = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X):
        clf = LogisticRegression().fit(X[train_idx], y[train_idx])
        loo_probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auc         = roc_auc_score(y, loo_probs)
    fpr, tpr, _ = roc_curve(y, loo_probs)

    # Clustering recovery; unsupervised grouping against CHIP labels
    cluster_labels = AgglomerativeClustering(n_clusters = 2).fit_predict(X)
    ari = adjusted_rand_score(y, cluster_labels)

    # Save results
    results = dict(
        operating_threshold = config.analysis.vessel_threshold,
        n_chip              = len(chip),                        n_non_chip          = len(non_chip),
        mean_chip           = chip.mean(),                      sd_chip             = chip.std(),
        mean_non_chip       = non_chip.mean(),                  sd_non_chip         = non_chip.std(),
        shapiro_p_chip      = shapiro_chip,                     shapiro_p_non_chip  = shapiro_non,       
        levene_p            = levene_p,
        welch_t             = t_stat,                           welch_p             = t_p,               
        mannwhitney_u       = u_stat,  
        mannwhitney_p       = u_p,
        hedges_g            = g,                                boot_ci_95          = [ci_low, ci_high],
        loocv_auc           = auc,                              cluster_ari         = ari
    )
    results_path = analysis_dir / "stats_results.json"
    json.dump(results, open(results_path, "w"), indent = 2, default = float)

    # Plot all key figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.boxplot(data = sample_density, x = "chip_status", 
                y = "sample_fibroblast_density", ax = axes[0], 
                showfliers = False)
    sns.swarmplot(data = sample_density, x = "chip_status", 
                  y = "sample_fibroblast_density", ax = axes[0], 
                  color = "k", size = 4)
    
    axes[0].set_xticklabels(["non-CHIP", "CHIP"])
    axes[0].set_title(f"Welch p={t_p:.3f}, MWU p={u_p:.3f}, g={g:.2f}")

    axes[1].plot(fpr, tpr, label=f"AUC={auc:.2f}")
    axes[1].plot([0,1],[0,1],"--",color="gray")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("LOOCV ROC")
    axes[1].legend()

    axes[2].scatter(sample_density.sample_fibroblast_density, cluster_labels, 
                    c=y, cmap = "coolwarm")
    axes[2].set_yticks([0,1])
    axes[2].set_yticklabels(["Cluster 0","Cluster 1"])
    axes[2].set_title(f"Cluster vs CHIP (ARI={ari:.2f})")

    plt.tight_layout()
    plt.savefig(analysis_dir / "primary_analysis.png", dpi = 300)

    log_footer()
    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Workflow Analysis.")
    parser.add_argument("--config_dir",   type = str, default = "configs/")
    parser.add_argument("--version",      type = str)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]