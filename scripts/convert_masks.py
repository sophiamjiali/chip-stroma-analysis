#!/usr/bin/env python3
"""Copy tile_*_mask.png files from nested sample_id dirs to a sanitized destination tree.

Usage:
    python copy_sanitize_masks.py <src_dir> <dst_dir>
"""
import argparse
import re
import shutil
from pathlib import Path

MASK_PATTERN = re.compile(r"^tile_(.*)_mask\.png$")


def sanitize(name: str) -> str:
    return name.replace(" ", "_")


def main(src_dir: Path, dst_dir: Path) -> None:
    n_copied = 0
    for mask_path in src_dir.rglob("tile_*_mask.png"):
        m = MASK_PATTERN.match(mask_path.name)
        if not m:
            continue
        sample_id = mask_path.parent.name
        sample_id_clean = sanitize(sample_id)
        out_dir = dst_dir / sample_id_clean
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"tile_{m.group(1)}_vessel_mask.png"
        shutil.copy2(mask_path, out_dir / out_name)
        n_copied += 1
    print(f"Copied {n_copied} mask files to {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    args = parser.parse_args()
    main(args.src_dir, args.dst_dir)