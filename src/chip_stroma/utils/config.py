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

def load_configs(pipeline   : Path, 
                 paths      : Path,
                 config_name: str | None = None,
                 frozen     : bool = False) -> Box: 
    "Loads all configurations nested by their file name."

    logger.info("=" * 50)
    logger.info("Step 01: Configurations")
    logger.info(f"- Pipeline : {pipeline.relative_to(Path.cwd())}")
    logger.info(f"- Paths    : {paths.relative_to(Path.cwd())}")
    logger.info("-" * 50)

    # If a sweep, override the configuration name with just 'sweep'
    name = config_name if config_name else pipeline.stem.split('_', 1)[1]

    config = {
        name   : load_config(pipeline),
        "paths": load_paths_config(paths)
    }

    logger.info("Successfully loaded and merged both configuration files")
    logger.info("=" * 50)

    return Box(config, frozen_box = frozen)
    

# =====| Helper Functions |=====================================================

def load_paths_config(paths: Path) -> dict:
    "Loads and resolves all paths in paths.yaml to the project root."

    # Extract environment constants from the .env file
    ROOT_CONSTS = extract_env()

    # Recursively resolve full paths of each nested path
    config = load_config(paths)
    config = resolve_paths(config, ROOT_CONSTS['project_root'])

    # Append the raw directories directly into the configurations
    config['raw_data'] = {
        "raw_dir"        : ROOT_CONSTS['raw_dir'],
        "patch_dir"      : ROOT_CONSTS['raw_patch_dir'],
        "vessel_mask_dir": ROOT_CONSTS['raw_vessel_mask_dir'],
        "coordinate_dir" : ROOT_CONSTS['patch_coords_dir']
    }

    return config

    
def extract_env() -> dict:
    "Safe loading for .env constants."

    load_dotenv(ROOT / ".env")
    PROJECT_ROOT        = os.getenv("PROJECT_ROOT", ".")
    RAW_DIR             = os.getenv("RAW_DIR", ".")
    RAW_PATCH_DIR       = os.getenv("RAW_PATCH_DIR", ".")
    RAW_VESSEL_MASK_DIR = os.getenv("RAW_VESSEL_MASK_DIR", ".")
    PATCH_COORDS_DIR    = os.getenv("PATCH_COORDS_DIR", ".")

    return {
        'project_root': PROJECT_ROOT,
        'raw_dir': RAW_DIR,
        'raw_patch_dir': RAW_PATCH_DIR,
        'raw_vessel_mask_dir': RAW_VESSEL_MASK_DIR,
        'patch_coords_dir': PATCH_COORDS_DIR
    }


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


def resolve_params(trial_params: Box, 
                   sweep_params: Box) -> Box:
    """
    Return a flat Box containing all leaf parameters from sweep_params,
    overridden by values from trial_params where present.
    """
    merged = Box()

    def flatten(node):
        for key, value in node.items():
            if isinstance(value, Box):
                flatten(value)
            else:
                merged[key] = trial_params.get(key, value)

    flatten(sweep_params)
    return merged

# [END]