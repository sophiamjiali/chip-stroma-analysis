# ==============================================================================
# Script:           05_multiseed.py
# Purpose:          Multi-seed configuration of top-K trials on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/21/2026
# ==============================================================================

import argparse as ap
import pandas as pd

from pathlib import Path
from copy import deepcopy
from box import Box

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.training.train import run_seed
from chip_stroma.utils.model_utils import get_top_k_trials

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Multi-Seed Confirmation",
        config_path    = Path(args.config_dir) / "05_multiseed.yaml",
        version        = args.version
    )

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "05_multiseed.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # 2. Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )

    # Extract the top K trials
    storage = f"sqlite:///{config.paths.studies}/{args.version}.db"
    top_trials = get_top_k_trials(
        storage = storage,
        version = args.version,
        k       = config.multiseed.top_k
    )

    group = f"{args.version}_multiseed"

    # Train each top trial the set amount of seeds
    results = []
    for trial in top_trials:
        trial_params = Box(deepcopy(trial.params), frozen = True)

        print(trial_params)

        for seed in range(int(config.multiseed.n_seeds)):
            results.append(run_seed(
                manifest        = manifest,
                project         = config.multiseed.project,
                group           = group,
                paths           = config.paths,
                trial_params    = trial_params,
                callback_params = config.multiseed.callbacks,
                fold            = trial_params['fold'],
                seed            = seed,
                trial_num       = trial.number
            ))

    # Aggregate mean and standard deviation per trial
    results = pd.DataFrame(results)
    summary = (results.groupby('trial_num')['best_val_dice']
               .agg(['mean', 'std', 'count']))

    summary_path = (config.paths.results / args.version / 
                    "multiseed_summary.csv")
    summary_path.parent.mkdir(parents = True, exist_ok = True)
    summary.to_csv(summary_path)

    log_footer()

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Run multi-seed confirmation.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version",    type = str)
    
    return parser.parse_args()


if __name__ == "__main__":
    main()