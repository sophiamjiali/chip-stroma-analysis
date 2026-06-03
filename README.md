# chip-stroma-analysis
Computational pipeline for automated segmentation and quantification of stromal remodeling in CHIP bone marrow biopsies using multiplex IHC staining


```
chip-stroma-analysis/
├── data/
│   ├── raw/                        # original WSIs/patches — gitignored
│   ├── masks/                      # region + vessel annotation masks — gitignored
│   └── processed/                  # normalized patches, processed masks — gitignored
├── configs/
│   ├── segmentation.yaml           # U-Net hyperparameters, augmentation, training
│   └── classification.yaml        # logistic regression hyperparameters
│   └── sweeps/
│       └── segmentation_sweep.yaml    # wandb sweep configuration
├── src/
│   └── chip_stroma/                    # main package
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py              # PyTorch Dataset for patches + masks
│       │   └── transforms.py          # stain normalization, augmentation
│       ├── segmentation/
│       │   ├── __init__.py
│       │   ├── model.py                # LightningModule wrapping SMP U-Net
│       │   ├── train.py                # trainer entrypoint
│       │   └── predict.py             # inference on new slides
│       ├── quantification/
│       │   ├── __init__.py
│       │   ├── area.py                 # compute aSMA+ region area per patient
│       │   └── vessel_exclusion.py    # subtract vessel annotations post-segmentation
│       └── classification/
│           ├── __init__.py
│           ├── logistic_regression.py  # patient-level CHIP vs non-CHIP
│           └── statistics.py          # t-test replication, effect size
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_validation.ipynb
│   └── 03_classification_results.ipynb
├── tests/
│   ├── test_dataset.py
│   ├── test_area.py
│   └── test_vessel_exclusion.py
├── scripts/
│   ├── preprocess.py              # patch extraction, mask alignment
│   ├── evaluate.py                # full pipeline evaluation
│   └── run_sweep.py               # wandb sweep agent entrypoint
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```