# ==============================================================================
# Script:           06_full_cv.py
# Purpose:          Multi-seed configuration of top-K trials on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/21/2026
# ==============================================================================

import argparse as ap
import pandas as pd

from pathlib import Path

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.utils.model_utils import select_final_config
from chip_stroma.training.train import run_seed

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Full Cross-Validation",
        config_path    = Path(args.config_dir) / "06_multiseed.yaml",
        version        = args.version
    )

    # 1. Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "06_full_cv.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    # 2. Verify that the training manifest was created, else create it
    manifest = initialize_train_manifest(
        train_manifest_path = config.paths.metadata.train_manifest,
        patch_manifest_path = config.paths.metadata.patch_manifest
    )

    # 3. Resolve the final configuration from multi-seed confirmation
    trial_params, trial_num = select_final_config(config, args.version)

    ckpt_dir = Path(config.paths.checkpoints.full_cv) / args.version
    group    = f"{args.version}_full_cv"

    # Train a model per fold; each fold each out in turn as validation
    results = []
    for fold in range(config.full_cv.n_folds):

        ckpt_path = ckpt_dir / f"fold_{fold}.ckpt"

        # Initialize the path for the fold's checkpoint
        results.append(run_seed(
            manifest        = manifest,
            project         = config.full_cv.project,
            group           = group,
            paths           = config.paths,
            trial_params    = trial_params,
            callback_params = config.full_cv.callbacks,
            fold            = fold,
            seed            = trial_params['seed'],
            trial_num       = trial_num,
            ckpt_path       = ckpt_path
        ))

    # Aggregate mean/std Dice across folds as an unbiased generalized estimate
    results = pd.DataFrame(results)
    summary = (results[['val_dice', 'val_precision', 'val_recall']]
               .agg(['mean', 'std']))
    summary['source_trial_num'] = trial_num
    
    summary_path = config.paths.results / args.version / "full_cv_summary.csv"
    summary_path.parent.mkdir(parents = True, exist_ok = True)
    summary.to_csv(summary_path)

    logger.info("Full CV complete")
    log_footer()

    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Run Full Cross-Validation.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version",    type = str)
    
    return parser.parse_args()


if __name__ == "__main__":
    main()