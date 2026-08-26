# Details

Date : 2026-08-26 16:35:40

Directory /Users/sophiali/Desktop/chip-stroma-analysis

Total : 95 files,  6540 codes, 1932 comments, 1741 blanks, all 10213 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 71 | 0 | 8 | 79 |
| [configs/00\_paths.yaml](/configs/00_paths.yaml) | YAML | 17 | 10 | 5 | 32 |
| [configs/01\_preprocess.yaml](/configs/01_preprocess.yaml) | YAML | 14 | 7 | 4 | 25 |
| [configs/02\_cross\_validation.yaml](/configs/02_cross_validation.yaml) | YAML | 5 | 7 | 1 | 13 |
| [configs/03\_train.yaml](/configs/03_train.yaml) | YAML | 37 | 8 | 8 | 53 |
| [configs/05\_multiseed.yaml](/configs/05_multiseed.yaml) | YAML | 14 | 7 | 6 | 27 |
| [configs/06\_full\_cv.yaml](/configs/06_full_cv.yaml) | YAML | 14 | 7 | 6 | 27 |
| [configs/07\_inference.yaml](/configs/07_inference.yaml) | YAML | 8 | 7 | 2 | 17 |
| [configs/08\_evaluate.yaml](/configs/08_evaluate.yaml) | YAML | 5 | 7 | 2 | 14 |
| [configs/09\_visualize.yaml](/configs/09_visualize.yaml) | YAML | 2 | 7 | 1 | 10 |
| [configs/10\_quantify.yaml](/configs/10_quantify.yaml) | YAML | 4 | 7 | 3 | 14 |
| [configs/11\_analysis.yaml](/configs/11_analysis.yaml) | YAML | 1 | 7 | 1 | 9 |
| [configs/pipeline.yaml](/configs/pipeline.yaml) | YAML | 14 | 8 | 2 | 24 |
| [configs/sweeps/template.yaml](/configs/sweeps/template.yaml) | YAML | 35 | 12 | 8 | 55 |
| [configs/sweeps/v1.yaml](/configs/sweeps/v1.yaml) | YAML | 31 | 12 | 7 | 50 |
| [configs/sweeps/v2.yaml](/configs/sweeps/v2.yaml) | YAML | 35 | 12 | 8 | 55 |
| [configs/sweeps/v3.yaml](/configs/sweeps/v3.yaml) | YAML | 36 | 12 | 8 | 56 |
| [configs/sweeps/v4.yaml](/configs/sweeps/v4.yaml) | YAML | 36 | 12 | 8 | 56 |
| [configs/sweeps/v5.yaml](/configs/sweeps/v5.yaml) | YAML | 36 | 12 | 8 | 56 |
| [configs/sweeps/v6.yaml](/configs/sweeps/v6.yaml) | YAML | 42 | 12 | 10 | 64 |
| [configs/sweeps/v7.yaml](/configs/sweeps/v7.yaml) | YAML | 42 | 12 | 10 | 64 |
| [configs/sweeps/v8.yaml](/configs/sweeps/v8.yaml) | YAML | 43 | 12 | 10 | 65 |
| [exploration/all\_patches\_positive\_counts.csv](/exploration/all_patches_positive_counts.csv) | CSV | 25 | 0 | 1 | 26 |
| [exploration/count\_positive\_counts.py](/exploration/count_positive_counts.py) | Python | 30 | 1 | 16 | 47 |
| [exploration/create\_svs\_thumbnails.py](/exploration/create_svs_thumbnails.py) | Python | 22 | 1 | 13 | 36 |
| [exploration/filtered\_positive\_counts.csv](/exploration/filtered_positive_counts.csv) | CSV | 25 | 0 | 1 | 26 |
| [notebooks/00\_view\_binary\_mask.ipynb](/notebooks/00_view_binary_mask.ipynb) | JSON | 77 | 0 | 1 | 78 |
| [notebooks/01\_convert\_mask\_to\_binary.ipynb](/notebooks/01_convert_mask_to_binary.ipynb) | JSON | 114 | 0 | 1 | 115 |
| [notebooks/02\_preprocess.ipynb](/notebooks/02_preprocess.ipynb) | JSON | 758 | 0 | 1 | 759 |
| [notebooks/03\_train.ipynb](/notebooks/03_train.ipynb) | JSON | 159 | 0 | 1 | 160 |
| [notebooks/04\_analyze\_sweep.ipynb](/notebooks/04_analyze_sweep.ipynb) | JSON | 145 | 0 | 1 | 146 |
| [notebooks/05\_evaluate.ipynb](/notebooks/05_evaluate.ipynb) | JSON | 173 | 0 | 1 | 174 |
| [notebooks/10\_visualize.ipynb](/notebooks/10_visualize.ipynb) | JSON | 63 | 0 | 1 | 64 |
| [requirements/requirements.txt](/requirements/requirements.txt) | pip requirements | 25 | 0 | 1 | 26 |
| [scripts/01\_preprocess.py](/scripts/01_preprocess.py) | Python | 106 | 21 | 27 | 154 |
| [scripts/02\_assign\_folds.py](/scripts/02_assign_folds.py) | Python | 50 | 15 | 21 | 86 |
| [scripts/03\_train.py](/scripts/03_train.py) | Python | 46 | 13 | 18 | 77 |
| [scripts/04a\_init\_study.py](/scripts/04a_init_study.py) | Python | 38 | 12 | 13 | 63 |
| [scripts/04b\_sweep.py](/scripts/04b_sweep.py) | Python | 102 | 15 | 30 | 147 |
| [scripts/05a\_multiseed.py](/scripts/05a_multiseed.py) | Python | 72 | 17 | 33 | 122 |
| [scripts/05b\_aggregate\_multiseed.py](/scripts/05b_aggregate_multiseed.py) | Python | 30 | 11 | 14 | 55 |
| [scripts/06a\_full\_cv.py](/scripts/06a_full_cv.py) | Python | 69 | 15 | 32 | 116 |
| [scripts/06b\_aggregate\_full\_cv.py](/scripts/06b_aggregate_full_cv.py) | Python | 30 | 11 | 14 | 55 |
| [scripts/07\_inference.py](/scripts/07_inference.py) | Python | 93 | 18 | 36 | 147 |
| [scripts/08\_evaluate.py](/scripts/08_evaluate.py) | Python | 97 | 18 | 29 | 144 |
| [scripts/09\_visualize.py](/scripts/09_visualize.py) | Python | 100 | 19 | 30 | 149 |
| [scripts/10\_quantify.py](/scripts/10_quantify.py) | Python | 77 | 16 | 22 | 115 |
| [scripts/11\_analysis.py](/scripts/11_analysis.py) | Python | 151 | 28 | 41 | 220 |
| [scripts/12\_stitch\_masks.py](/scripts/12_stitch_masks.py) | Python | 112 | 20 | 37 | 169 |
| [slurm/00\_setup\_env.sh](/slurm/00_setup_env.sh) | Shell Script | 9 | 5 | 8 | 22 |
| [slurm/01\_run\_preprocess.sh](/slurm/01_run_preprocess.sh) | Shell Script | 11 | 10 | 6 | 27 |
| [slurm/02\_run\_fold\_assignment.sh](/slurm/02_run_fold_assignment.sh) | Shell Script | 11 | 10 | 6 | 27 |
| [slurm/03\_run\_train.sh](/slurm/03_run_train.sh) | Shell Script | 16 | 14 | 8 | 38 |
| [slurm/04\_run\_sweep.sh](/slurm/04_run_sweep.sh) | Shell Script | 39 | 15 | 13 | 67 |
| [slurm/04a\_init\_study.sh](/slurm/04a_init_study.sh) | Shell Script | 15 | 9 | 8 | 32 |
| [slurm/04b\_submit\_sweep.sh](/slurm/04b_submit_sweep.sh) | Shell Script | 20 | 14 | 9 | 43 |
| [slurm/05\_run\_multiseed.sh](/slurm/05_run_multiseed.sh) | Shell Script | 42 | 15 | 15 | 72 |
| [slurm/05a\_submit\_multiseed.sh](/slurm/05a_submit_multiseed.sh) | Shell Script | 22 | 14 | 10 | 46 |
| [slurm/05b\_aggregate\_multiseed.sh](/slurm/05b_aggregate_multiseed.sh) | Shell Script | 13 | 10 | 7 | 30 |
| [slurm/06\_run\_full\_cv.sh](/slurm/06_run_full_cv.sh) | Shell Script | 40 | 15 | 14 | 69 |
| [slurm/06a\_submit\_full\_cv.sh](/slurm/06a_submit_full_cv.sh) | Shell Script | 22 | 14 | 10 | 46 |
| [slurm/06b\_aggregate\_full\_cv.sh](/slurm/06b_aggregate_full_cv.sh) | Shell Script | 13 | 10 | 7 | 30 |
| [slurm/07\_run\_inference.sh](/slurm/07_run_inference.sh) | Shell Script | 13 | 13 | 6 | 32 |
| [slurm/08\_run\_evaluate.sh](/slurm/08_run_evaluate.sh) | Shell Script | 13 | 13 | 6 | 32 |
| [slurm/09\_run\_visualize.sh](/slurm/09_run_visualize.sh) | Shell Script | 13 | 13 | 6 | 32 |
| [slurm/10\_run\_quantify.sh](/slurm/10_run_quantify.sh) | Shell Script | 13 | 13 | 6 | 32 |
| [slurm/11\_run\_analysis.sh](/slurm/11_run_analysis.sh) | Shell Script | 12 | 10 | 6 | 28 |
| [slurm/submit\_job.sh](/slurm/submit_job.sh) | Shell Script | 32 | 18 | 11 | 61 |
| [slurm/sync\_wandb.sh](/slurm/sync_wandb.sh) | Shell Script | 9 | 4 | 4 | 17 |
| [src/chip\_stroma/classification/logistic\_regression.py](/src/chip_stroma/classification/logistic_regression.py) | Python | 2 | 8 | 1 | 11 |
| [src/chip\_stroma/classification/statistics.py](/src/chip_stroma/classification/statistics.py) | Python | 45 | 18 | 17 | 80 |
| [src/chip\_stroma/data/datamodule.py](/src/chip_stroma/data/datamodule.py) | Python | 91 | 82 | 29 | 202 |
| [src/chip\_stroma/data/dataset.py](/src/chip_stroma/data/dataset.py) | Python | 45 | 39 | 18 | 102 |
| [src/chip\_stroma/data/preprocessing.py](/src/chip_stroma/data/preprocessing.py) | Python | 391 | 288 | 116 | 795 |
| [src/chip\_stroma/data/sampler.py](/src/chip_stroma/data/sampler.py) | Python | 44 | 38 | 19 | 101 |
| [src/chip\_stroma/data/transforms.py](/src/chip_stroma/data/transforms.py) | Python | 22 | 42 | 12 | 76 |
| [src/chip\_stroma/evaluate/segmentation\_stats.py](/src/chip_stroma/evaluate/segmentation_stats.py) | Python | 132 | 36 | 50 | 218 |
| [src/chip\_stroma/models/inference.py](/src/chip_stroma/models/inference.py) | Python | 77 | 24 | 25 | 126 |
| [src/chip\_stroma/models/loss.py](/src/chip_stroma/models/loss.py) | Python | 100 | 60 | 38 | 198 |
| [src/chip\_stroma/models/model.py](/src/chip_stroma/models/model.py) | Python | 283 | 57 | 87 | 427 |
| [src/chip\_stroma/quantification/density\_score.py](/src/chip_stroma/quantification/density_score.py) | Python | 150 | 29 | 43 | 222 |
| [src/chip\_stroma/training/create\_study.py](/src/chip_stroma/training/create_study.py) | Python | 56 | 11 | 26 | 93 |
| [src/chip\_stroma/training/cross\_validation.py](/src/chip_stroma/training/cross_validation.py) | Python | 98 | 79 | 40 | 217 |
| [src/chip\_stroma/training/objective.py](/src/chip_stroma/training/objective.py) | Python | 125 | 50 | 44 | 219 |
| [src/chip\_stroma/training/predict.py](/src/chip_stroma/training/predict.py) | Python | 0 | 7 | 0 | 7 |
| [src/chip\_stroma/training/train.py](/src/chip_stroma/training/train.py) | Python | 207 | 38 | 43 | 288 |
| [src/chip\_stroma/utils/callbacks.py](/src/chip_stroma/utils/callbacks.py) | Python | 115 | 44 | 43 | 202 |
| [src/chip\_stroma/utils/config.py](/src/chip_stroma/utils/config.py) | Python | 77 | 18 | 31 | 126 |
| [src/chip\_stroma/utils/header\_footers.py](/src/chip_stroma/utils/header_footers.py) | Python | 18 | 10 | 9 | 37 |
| [src/chip\_stroma/utils/io.py](/src/chip_stroma/utils/io.py) | Python | 283 | 79 | 143 | 505 |
| [src/chip\_stroma/utils/loggers.py](/src/chip_stroma/utils/loggers.py) | Python | 57 | 22 | 21 | 100 |
| [src/chip\_stroma/utils/model\_utils.py](/src/chip_stroma/utils/model_utils.py) | Python | 46 | 23 | 24 | 93 |
| [src/chip\_stroma/visualize/overlays.py](/src/chip_stroma/visualize/overlays.py) | Python | 115 | 42 | 50 | 207 |
| [src/chip\_stroma/visualize/quantification\_plots.py](/src/chip_stroma/visualize/quantification_plots.py) | Python | 76 | 8 | 26 | 110 |
| [src/chip\_stroma/visualize/segmentation\_plots.py](/src/chip_stroma/visualize/segmentation_plots.py) | Python | 176 | 93 | 58 | 327 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)