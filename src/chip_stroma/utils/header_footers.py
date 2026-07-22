# ==============================================================================
# Script:           header_footers.py
# Purpose:          General header and footers for entry-point scripts
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/22/2026
# ==============================================================================

from datetime import datetime
from pathlib import Path

from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


def log_header(pipeline_stage: str, config_path: Path, version = None):
    """Header for entry-point script."""

    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info(f"- Pipeline Stage: {pipeline_stage}")
    logger.info(f"- Configurations: {config_path}")
    if version is not None: logger.info(f"- Version: {version}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

def log_footer():
    """Footer for entry-point script."""

    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

# [END]