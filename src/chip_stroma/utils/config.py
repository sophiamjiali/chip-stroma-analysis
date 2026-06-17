# ==============================================================================
# Script:           paths.yaml
# Purpose:          Configuration utilities
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import os
import yaml
import logging

from pathlib import Path
from box import Box
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)

# =====| Main API |=============================================================

def load_configs(pipeline: Path, paths: Path) -> Box:
    "Loads all configurations nested by their file name."

    logger.info("=" * 50)
    logger.info("Step 01: Configurations")
    logger.info(f"- Pipeline: {pipeline}")
    logger.info(f"- Paths: {paths}")
    logger.info("-" * 50)

    config = {
        pipeline.stem: load_config(pipeline),
        "paths": load_paths_config(paths)
    }

    logger.info("Successfully loaded and merged both configuration files")
    logger.info("=" * 50)

    return Box(config, frozen_box = True)
    

# =====| Helper Functions |=====================================================

def load_paths_config(paths: Path) -> dict:
    "Loads and resolves all paths in paths.yaml to the project root."

    # Extract environment constants from the .env file
    PROJECT_ROOT, RAW_DIR, RAW_PATCH_DIR, RAW_VESSEL_MASK_DIR = extract_env()

    # Recursively resolve full paths of each nested path
    config = load_config(paths)
    config = resolve_paths(config, PROJECT_ROOT)

    # Append the raw directories directly into the configurations
    config['raw_data'] = {
        "raw_dir": RAW_DIR,
        "patch_dir": RAW_PATCH_DIR,
        "vessel_mask_dir": RAW_VESSEL_MASK_DIR
    }

    return config

    
def extract_env() -> tuple[Path, Path, Path, Path]:
    "Safe loading for .env constants."

    load_dotenv(ROOT / ".env")
    PROJECT_ROOT        = os.getenv("PROJECT_ROOT", ".")
    RAW_DIR             = os.getenv("RAW_DIR", ".")
    RAW_PATCH_DIR       = os.getenv("RAW_PATCH_DIR", ".")
    RAW_VESSEL_MASK_DIR = os.getenv("RAW_VESSEL_MASK_DIR", ".")
    
    return (Path(PROJECT_ROOT), Path(RAW_DIR), 
            Path(RAW_PATCH_DIR), Path(RAW_VESSEL_MASK_DIR))


def resolve_paths(data, root) -> dict:
    "Recursively resolves string paths in a dictionary to absolute Path objects"

    resolved = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = resolve_paths(value, root)
        elif isinstance(value, str):
            resolved[key] = root / value
        else:
            resolved[key] = value
    return resolved


def load_config(path: Path) -> dict:
    return yaml.safe_load(open(path))

# [END]