import argparse as ap
import pandas as pd
from pathlib import Path
from chip_stroma.utils.config import load_configs

def main():
    args = parse_args()
    config = load_configs(
        pipeline = Path(args.config_dir) / "05_multiseed.yaml",
        paths    = Path(args.config_dir) / "00_paths.yaml"
    )
    task_dir = config.paths.results / args.version / "multiseed_tasks"
    results = pd.concat([pd.read_csv(f) for f in task_dir.glob("*.csv")], ignore_index=True)

    summary = (results.groupby('trial_num')['best_val_dice']
                       .agg(['mean', 'std', 'count']))
    summary_path = config.paths.results / args.version / "multiseed_summary.csv"
    summary.to_csv(summary_path)

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs/")
    parser.add_argument("--version", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    main()