# ==============================================================================
# Script:           04a_sweep.py
# Purpose:          Initializes Optuna SQL database for sweep SLURM array task
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/27/2026
# ==============================================================================

from pathlib import Path
import argparse as ap

from chip_stroma.utils.loggers import setup_logger
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.training.create_study import initialize_study

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================
def main():
    args = parse_args()
    pipeline_path = Path(args.config_dir) / "sweeps" / f"{args.version}.yaml"
    log_header(
        pipeline_stage = "Study Initialization",
        config_path    = pipeline_path,
        version        = args.version
    )

    # 1. Load workflow and path configurations; sweeps are nested in a folder
    config = load_configs(
        pipeline    = pipeline_path,
        paths       = Path(args.config_dir) / "00_paths.yaml",
        config_name = "sweep",
        frozen      = False
    )

    # Initialize the study to avoid SLURM array race conditions
    _ = initialize_study(
        version          = args.version,
        seed             = config.sweep.data.seed,
        n_startup_trials = config.sweep.experiment.n_startup_trials,
        n_warmup_steps   = config.sweep.experiment.n_warmup_steps,
        studies_dir      = config.paths.studies
    )

    logger.info("=" * 50)
    log_footer()
    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Begin a hyperparameter sweep.")
    parser.add_argument("--config_dir", type = str, default = "configs/sweeps/")
    parser.add_argument("--version",    type = str, default = "v1")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]