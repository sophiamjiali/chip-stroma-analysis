# ==============================================================================
# Script:           logger.py
# Purpose:          Initializes logger output
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import logging
import sys

def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Create or retrieve a logger with safe default configuration. Enforces
    absolute determinism for logging and does not depend on upstream
    logging configuration.

    Ensures logging works in scripts and SLURM environments where
    no prior logging configuration exists.
    """

    logging.basicConfig(
        level=logging.INFO,
        format   = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers = [logging.StreamHandler(sys.stdout)],
        force    = True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    return logger

# [END]