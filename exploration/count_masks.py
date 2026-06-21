#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def has_positive_pixel(png_path: Path) -> tuple[str, bool]:
    mask = np.array(Image.open(png_path))
    return png_path.parts[-2], bool(np.any(mask > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str)
    parser.add_argument("--out_csv", type=str, default="positive_mask_counts.csv")
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    # collect all png paths (single lightweight pass)
    png_paths = list(root_dir.rglob("*.png"))

    counts = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sample_id, has_pos in tqdm(
            pool.map(has_positive_pixel, png_paths, chunksize=500),
            total=len(png_paths)
        ):
            if sample_id not in counts:
                counts[sample_id] = 0
            if has_pos:
                counts[sample_id] += 1

    df = pd.DataFrame(
        [(k, v) for k, v in counts.items()],
        columns=["sample_id", "positive_png_count"]
    )

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()