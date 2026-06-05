# ==============================================================================
# Script:           paths.yaml
# Purpose:          Input and output utilities with sanitization
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/04/2026
# ==============================================================================

import shutil

import numpy as np

from pathlib import Path
from PIL import Image
from typing import Callable

# =====| Preprocessing Step Wrappers |==========================================

def run_step(src_dir: Path,
             dst_dir: Path,
             process_fn: Callable,
             cleanup_src: bool = False,
             **kwargs) -> None:
    """
    Wrapper function to perform a preprocessing step on aSMA stain patches 
    individually. Extracts patches from the raw/intermediate source directory
    and saves it in the intermediate/processed destination directory.

    If the cleanup is toggled, the source directory is deleted.
    """

    # Ensure the preceeding step was completed appropriately
    assert step_complete(src_dir), f"The provided source directory was not marked as complete: {src_dir}"

    # Initialize the destination folders; patches and masks respectively
    dst_patch_dir = dst_dir / "patches"
    dst_mask_dir = dst_dir / "masks"

    dst_patch_dir.mkdir(parents = True, exist_ok = True)
    dst_mask_dir.mkdir(parents = True, exist_ok = True)
    
    # Extract the source folders
    src_patch_dir = src_dir / "patches"
    src_mask_dir = src_dir / "masks"

    # Load, process, and save each patch individually
    for patch_path in src_patch_dir.rglob("*.png"):

        # Preserve patient subdirectory structure
        relative = patch_path.relative_to(src_patch_dir)
        dst_patch = dst_patch_dir / relative
        dst_patch.parent.mkdir(parents = True, exist_ok = True)

        patch = np.array(Image.open(patch_path))
        result = process_fn(patch, **kwargs)

        # Patch passed; copy patch and mask to destination
        if result is not None:
            Image.fromarray(result).save(dst_patch)

            # Copy the mask into the destination directory
            mask_name = patch_path.name.replace("_raw.png", "_mask.png")
            mask_path = src_mask_dir / relative.parent / mask_name
            dst_mask = dst_mask_dir / relative.parent / mask_name
            dst_mask.parent.mkdir(parents = True, exist_ok = True)

            assert mask_path.exists(), f"Warning: missing mask for {patch_path}"
            shutil.copy(mask_path, dst_mask)

    mark_complete(dst_dir)

    # Delete the source directories if toggled
    if cleanup_src:
        assert step_complete(dst_dir), "Step did not complete -- aborting"
        shutil.rmtree(src_dir)
        shutil.rmtree(src_mask_dir)

    return


def mark_complete(directory: Path) -> None:
    (directory / "patches" / ".complete").touch()
    (directory / "masks" / ".complete").touch()


def step_complete(directory: Path) -> bool:
    return ((directory / "patches" / ".complete").exists() and
            (directory / "masks" / ".complete").exists())


def move_directory_contents(src_dir: Path, dst_dir: Path) -> None:
    "Copy the contents of the source directory into the destination."

    assert src_dir.exists(), f"Source directory {src_dir} was not found."
    dst_dir.mkdir(parents = True, exist_ok = True)

    # Copy over all files and their nested structures to the destination
    for item in src_dir.rglob("*"):
        relative_path = item.relative_to(src_dir)
        target_path = dst_dir / relative_path
        
        if item.is_dir():
            target_path.mkdir(parents = True, exist_ok = True)
        else:
            target_path.parent.mkdir(parents = True, exist_ok = True)
            shutil.move(str(item), str(target_path))

    # Remove the empty src_dir after moving everything
    shutil.rmtree(src_dir)

    return
    


# =====| Preprocessing Utility Functions |======================================

def sanitize_names(src_dir: Path, 
                   out_dir: Path) -> None:
    return None
    


def save_name_mapping(name_mapping: str, path: Path) -> None:
    return None

# [END]