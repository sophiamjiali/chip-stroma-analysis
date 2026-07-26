# ==============================================================================
# Script:           05b_aggregate_multiseed.py
# Purpose:          Multi-seed configuration of top-K trials on validation fold
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/24/2026
# ==============================================================================

import argparse as ap
import pandas as pd
from pathlib import Path

from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    config = load_configs(
        pipeline = Path(args.config_dir) / "05_multiseed.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    logger.info("Beginning multiseed aggregation")

    # Load all trial summary tables
    task_dir = config.paths.results / args.version / "multiseed_tasks"
    results  = pd.concat([pd.read_csv(f) for f in task_dir.glob("*.csv")], 
                         ignore_index = True)

    logger.info(f"Identified {len(results)} multi-seed summaries")

    # Summarize into one aggregate
    summary = (results.groupby('trial_num')['best_val_dice']
                       .agg(['mean', 'std', 'count']))
    summary_path = config.paths.results / args.version / "multiseed_summary.csv"
    summary_path.parent.mkdir(parents = True, exist_ok = True)
    summary.to_csv(summary_path)

    logger.info("Completed multiseed aggregation")

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version",    type = str)
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]