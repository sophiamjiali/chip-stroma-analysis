#!/usr/bin/env python3
"""
eval_haem_cv_distribution.py

Compute and plot the haematoxylin CV distribution across all patches
in the patch manifest. Intended for HPC execution to diagnose tissue
detection thresholds on the full dataset.

Usage
-----
    ./eval_haem_cv_distribution.py \
        --patch-dir  /path/to/patches \
        --manifest   /path/to/patch_manifest.csv \
        --output-dir /path/to/outputs \
        [--n-workers 8]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib    import Path
from PIL        import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.color      import separate_stains, hdx_from_rgb
from tqdm               import tqdm


# ---------------------------------------------------------------------------
# Per-patch worker (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _compute_cv(args: tuple) -> dict:
    """Compute haematoxylin CV for a single patch."""
    patch_path, original_id, patch_name = args
    try:
        patch  = np.array(Image.open(patch_path).convert("RGB"))
        stains = separate_stains(patch / 255.0, hdx_from_rgb)
        haem   = np.clip(stains[:, :, 0], 0, None)
        cv     = haem.std() / (haem.mean() + 1e-8)
        return {"original_id": original_id, "patch": patch_name,
                "haem_cv": cv, "error": None}
    except Exception as e:
        return {"original_id": original_id, "patch": patch_name,
                "haem_cv": np.nan, "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute haematoxylin CV distribution across all patches."
    )
    parser.add_argument("--patch-dir",  type=Path, required=True,
                        help="Root patch directory (src_dir/{original_id}/*.png)")
    parser.add_argument("--manifest",   type=Path, required=True,
                        help="Path to patch_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Directory to save plot and results CSV")
    parser.add_argument("--n-workers",  type=int,  default=1,
                        help="Number of parallel workers (default: 4)")
    return parser.parse_args()


def compute_cv_distribution(
    manifest:  pd.DataFrame,
    patch_dir: Path,
    n_workers: int = 4,
) -> pd.DataFrame:

    args_list = [
        (patch_dir / row["original_id"] / row["patch"],
         row["original_id"],
         row["patch"])
        for _, row in manifest.iterrows()
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_cv, a): a for a in args_list}
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Computing haematoxylin CV"):
            results.append(future.result())

    df = pd.DataFrame(results)

    n_errors = df["error"].notna().sum()
    if n_errors > 0:
        print(f"[WARNING] {n_errors} patches failed — see error column in CSV")

    return df


def plot_distribution(cv_series: pd.Series, output_dir: Path) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(cv_series.dropna(), bins=80, edgecolor="none", color="steelblue")
    axes[0].set_xlabel("Haematoxylin CV")
    axes[0].set_ylabel("Patch Count")
    axes[0].set_title("Haematoxylin CV Distribution")
    for p, ls in [(0.25, "--"), (0.50, "-"), (0.75, ":")]:
        val = cv_series.quantile(p)
        axes[0].axvline(val, color="firebrick", linestyle=ls, linewidth=1.2,
                        label=f"p{int(p*100)}={val:.2f}")
    axes[0].legend(fontsize=8)

    # CDF
    sorted_cv = np.sort(cv_series.dropna())
    cdf       = np.arange(1, len(sorted_cv) + 1) / len(sorted_cv)
    axes[1].plot(sorted_cv, cdf, color="steelblue", linewidth=1.5)
    axes[1].set_xlabel("Haematoxylin CV threshold")
    axes[1].set_ylabel("Fraction of patches rejected")
    axes[1].set_title("CDF — Fraction Rejected vs Threshold")
    axes[1].grid(True, alpha=0.3)

    # Annotate descriptive stats on histogram
    stats = cv_series.describe()
    stats_text = (f"n={int(stats['count'])}  "
                  f"mean={stats['mean']:.3f}  "
                  f"median={cv_series.median():.3f}  "
                  f"std={stats['std']:.3f}  "
                  f"p25={stats['25%']:.3f}  "
                  f"p75={stats['75%']:.3f}")
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    save_path = output_dir / "haem_cv_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    print(f"Total patches: {len(manifest)}")

    df = compute_cv_distribution(manifest, args.patch_dir, args.n_workers)

    # Save results CSV
    csv_path = args.output_dir / "haem_cv_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Print summary
    cv_series = df["haem_cv"].dropna()
    print("\n--- Haematoxylin CV Summary ---")
    print(cv_series.describe().round(4).to_string())

    # Plot
    plot_distribution(cv_series, args.output_dir)


if __name__ == "__main__":
    main()