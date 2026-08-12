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

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.classification.statistics import (
    hedges_g, 
    bootstrap_ci,
    run_loocv_auc
)
from chip_stroma.visualize.quantification_plots import (
    plot_chip_boxplot,
    plot_kde_overlap,
    plot_loocv_distribution, 
    plot_loocv_roc_curve
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

        label_dir = analysis_dir / label
        label_dir.mkdir(parents = True, exist_ok = True)

        # Plot key metric vs. CHIP status plot
        plot_chip_boxplot(
            sample_density = sample_density,
            col            = col,
            welsh_p        = t_p,
            mwu_p          = u_p,
            eff_g          = eff_g,
            out_path       = label_dir / f"boxplot_chip_status.png"
        )
        logger.info("- Saved boxplot")

        # Plot logistic regression
        plot_loocv_roc_curve(
            loocv    = loocv,
            out_path = label_dir / f"loocv_roc_curve.png"
        )
        logger.info("- Saved LOOCV ROC curve")
        
        # Plot KDE overlap panel
        plot_kde_overlap(
            sample_density = sample_density,
            col            = col,
            chip           = chip,
            non_chip       = non_chip,
            out_path       = label_dir / f"kde_overlap.png"
        )
        logger.info("- Saved KDE Overlap")

        # Plot LOOCV distribution
        plot_loocv_distribution(
            loocv    = loocv,
            y        = y,
            out_path = label_dir / f"loocv_distribution.png"
        )
        logger.info("- Saved LOOCV distribution")


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