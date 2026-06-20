#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def count_positive_masks(root_dir: Path) -> pd.DataFrame:
    rows = []

    root_dir = Path(root_dir)

    for sample_dir in root_dir.iterdir():
        if not sample_dir.is_dir():
            continue

        positive_count = 0

        for png_path in sample_dir.rglob("*.png"):
            mask = np.array(Image.open(png_path))

            if np.any(mask > 0):
                positive_count += 1

        rows.append({
            "sample_id": sample_dir.name,
            "positive_png_count": positive_count
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="Path to root directory containing mask folders")
    parser.add_argument("--out_csv", type=str, default="positive_mask_counts.csv")

    args = parser.parse_args()

    df = count_positive_masks(Path(args.root_dir))
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()