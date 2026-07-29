# ==============================================================================
# Script:           06a_full_cv.py
# Purpose:          Multi-seed configuration of top-K trials on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/21/2026
# ==============================================================================

import pandas as pd
import argparse as ap

from box import Box
from pathlib import Path

from chip_stroma.training.train import run_seed
from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.io import initialize_train_manifest
from chip_stroma.utils.model_utils import select_final_config
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs, load_config, resolve_params

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Full Cross-Validation",
        config_path    = Path(args.config_dir) / "06_multiseed.yaml",
        version        = args.version
    )

    logger.info("=" * 50)
    logger.info(f"Beginning fold: {args.task_id}")
    logger.info("=" * 50)

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
    sweep_path   = Path(args.config_dir) / "sweeps" / f"{args.version}.yaml"
    sweep_params = Box(load_config(sweep_path), frozen_box = True)

    trial_params = resolve_params(trial_params, sweep_params)




    # TEMPORARY
    trial_params['nsd_tolerance'] = 2.5





    ckpt_dir = Path(config.paths.checkpoints.full_cv) / args.version
    ckpt_dir.mkdir(parents = True, exist_ok = True)

    # Submit model training in a SLURM array
    fold   = args.task_id
    result = run_seed(
        manifest        = manifest,
        project         = config.full_cv.project,
        group           = f"{args.version}_full_cv",
        paths           = config.paths,
        trial_params    = trial_params,
        callback_params = config.full_cv.callbacks,
        fold            = fold,
        seed            = trial_params['seed'],
        trial_num       = trial_num,
        ckpt_path       = ckpt_dir / f"fold_{fold}.ckpt"
    )

    # Aggregate mean/std Dice across folds as an unbiased generalized estimate
    result = pd.DataFrame(result)
    out_dir = config.paths.results / args.version / "full_cv_tasks"
    out_dir.mkdir(parents = True, exist_ok = True)

    result.to_csv(out_dir / f"fold_{fold}_summary.csv", index = False)

    logger.info("=" * 50)
    logger.info(f"Completed fold: {args.task_id}")
    logger.info("=" * 50)

    log_footer()
    return


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Run Full Cross-Validation.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version",    type = str, default = "v0")
    parser.add_argument("--task_id",    type = int, default = None)
    
    return parser.parse_args()


if __name__ == "__main__":
    main()