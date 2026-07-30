# ==============================================================================
# Script:           10_quantify.yaml
# Purpose:          Quantify fibroblasts in aSMA stains
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/30/2026
# ==============================================================================

import argparse as ap

from pathlib import Path

from chip_stroma.utils.header_footers import log_header, log_footer
from chip_stroma.utils.config import load_configs
from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)

# =====| Workflow Entry Point |=================================================

def main():
    args = parse_args()
    log_header(
        pipeline_stage = "Quantification",
        config_path    = Path(args.config_dir) / "10_quantify.yaml",
        version        = args.version
    )

    # Load workflow and path configurations
    config = load_configs(
        pipeline = Path(args.config_dir) / "10_quantification.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )

    




    return

# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Fibroblast quantification.")
    parser.add_argument("--config_dir", type = str, default = "configs/")
    parser.add_argument("--version", type = str)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

# [END]