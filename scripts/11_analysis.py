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

from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.classification.statistics import (
    hedges_g, 
    bootstrap_ci,
    run_loocv_auc
)

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
    sample_density = sample_density.reset_index(drop = True)
    logger.info(f"- Read {len(sample_density)} sample-level fibroblast "
                f"density scores")


    # Run full statistics on all metrics
    metrics = {
        "fibroblast_density": "sample_fibroblast_density",
        "object_density": "sample_object_density", 
    }
    
    all_results = {}
    for label, col in metrics.items():

        logger.info(f"Beginning to process metric: {label}")

        # Extract CHIP and non-CHIP data
        chip     = (sample_density.loc[sample_density['chip_status'] == 1, col]
                    .dropna())
        non_chip = (sample_density.loc[sample_density['chip_status'] == 0, col]
                    .dropna())

        logger.info(f"- Identified {len(chip)} CHIP patients")
        logger.info(f"- Identified {len(non_chip)} non-CHIP patients")

        # Assumption check
        shapiro_chip = stats.shapiro(chip)[1]
        shapiro_non  = stats.shapiro(non_chip)[1]
        levene_p     = stats.levene(chip, non_chip)[1]
        logger.info("- Performed Shapiro-Wilk test")
        logger.info("- Performed Levene test")

        # Primary significance test
        t_stat, t_p = stats.ttest_ind(chip, non_chip, equal_var = False)
        logger.info("- Performed T-Test")

        # Non-parametric backup to T-Test
        u_stat, u_p = stats.mannwhitneyu(chip, non_chip, 
                                         alternative = "two-sided")
        logger.info("- Performed Mann-Whitney U test")

        # Effect size
        eff_g = hedges_g(chip, non_chip)
        logger.info("- Computed effect size")

        # Bootstrap CI on mean difference
        ci_low, ci_high = bootstrap_ci(chip.values, non_chip.values)
        logger.info("- Computed CI bootstraps on mean difference")

        # Separation via LOOCV logistic regression and AUC
        X = sample_density[[col]].dropna().values
        y = (sample_density.loc[sample_density[col].notna(), 
                                'chip_status'].values)
        
        loocv = run_loocv_auc(X, y)
        logger.info("- Evaluated CHIP vs. non-CHIP separation via OOCV "
                    "logistic regression")

        # Save results
        all_results[label] = dict(
            n_chip             = len(chip),
            n_non_chip         = len(non_chip),
            mean_chip          = chip.mean(),
            sd_chip            = chip.std(),
            mean_non_chip      = non_chip.mean(),
            sd_non_chip        = non_chip.std(),
            shapiro_p_chip     = shapiro_chip,
            shapiro_p_non_chip = shapiro_non,
            levene_p           = levene_p,
            welch_t            = t_stat,
            welch_p            = t_p,
            mannwhitney_u      = u_stat,                 
            mannwhitney_p      = u_p,
            hedges_g           = eff_g,
            boot_ci_95         = [ci_low, ci_high],
            loocv_auc          = loocv["auc"],
            auc_permutation_p  = loocv["perm_p"],
            youden_cutoff      = loocv["youden_cutoff"],
            sensitivity        = loocv["sens"],
            specificity        = loocv["spec"]
        )

        logger.info("- Finished computing all key statistics")

        # Plot key metric vs. CHIP status plot
        fig, axes = plt.subplots(1, 3, figsize = (15, 4))

        sns.boxplot(data = sample_density, x = "chip_status", 
                    y = col, ax = axes[0], showfliers = False)
        sns.swarmplot(data = sample_density, x = "chip_status", 
                      y = col, ax = axes[0], color = "k", size = 4)
        axes[0].set_xticklabels(["non-CHIP", "CHIP"])
        axes[0].set_title(f"Welch p = {t_p:.3f}, MWU p = {u_p:.3f}, "
                          f"g = {eff_g:.2f}")

        # Plot logistic regression
        axes[1].plot(loocv['fpr'], loocv['tpr'], 
                     label = f"AUC = {loocv['auc']:.2f}"
                     f"\nperm p = {loocv['perm_p']:.3f}")
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].set_title("LOOCV ROC")
        axes[1].legend()

        # Plot KDE overlap panel
        xs = np.linspace(sample_density[col].min(), 
                         sample_density[col].max(), 200)
        kde_g  = gaussian_kde(chip)(xs)
        kde_ng = gaussian_kde(non_chip)(xs)

        axes[2].plot(xs, kde_g,  label = "CHIP",     color = "firebrick")
        axes[2].plot(xs, kde_ng, label = "non-CHIP", color = "steelblue")
        axes[2].fill_between(xs, np.minimum(kde_g, kde_ng), alpha = 0.3, 
                             color = "gray", label = "overlap")
        axes[2].axvline(loocv["youden_cutoff"], linestyle="--", 
                        color = "k", label = "Youden cutoff")
         
        axes[2].set_title(f"Distribution overlap (sens = {loocv['sens']:.2f}, "
                          f"spec = {loocv['spec']:.2f})")
        axes[2].legend(fontsize = 8)

        plt.tight_layout()
        plt.savefig(analysis_dir / f"primary_analysis_{label}.png", dpi = 300)
        plt.close(fig)

        logger.info("- Saved key figure")


    results_path = analysis_dir / "stats_results.json"
    json.dump(all_results, open(results_path, "w"), indent = 2, default = float)

    logger.info("Completed all computations")

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

# [END]g