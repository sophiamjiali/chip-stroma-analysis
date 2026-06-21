#!/usr/bin/env python3

from pathlib import Path
import argparse

import openslide
from PIL import Image


def svs_to_thumbnail(svs_path: Path, out_path: Path) -> None:
    slide = openslide.OpenSlide(str(svs_path))

    thumbnail = slide.get_thumbnail(slide.level_dimensions[-1]).convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing .svs files")
    parser.add_argument("output_dir", type=str, help="Directory to save thumbnails")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    svs_files = list(input_dir.rglob("*.svs"))

    for svs_path in svs_files:
        out_path = output_dir / svs_path.with_suffix(".png").name
        svs_to_thumbnail(svs_path, out_path)


if __name__ == "__main__":
    main()