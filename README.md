# chip-stroma-analysis

Computational pipeline for automated quantification of stromal remodeling in CHIP bone marrow biopsies.

## Overview
Clonal Hematopoiesis of Indeterminate Potential (CHIP) is associated with fibroblast expansion and 
increased extracellular matrix production in the bone marrow. This pipeline automates quantification 
of αSMA+ fibroblast density from multiplex IHC staining (αSMA, Masson's Trichrome, Reticulin) to 
characterize the tumour microenvironment in CHIP versus non-CHIP patients.

## Pipeline
1. **Vessel Segmentation** — U-Net trained on vessel annotations to remove vascular regions
2. **Fibroblast Quantification** — Stain deconvolution and intensity thresholding on vessel-excluded regions
3. **Density Scoring** — αSMA+ region area per patient
4. **Statistical Validation** — t-test replication of CHIP vs non-CHIP fibroblast density difference
5. **Pathologist Validation** — export segmentation masks as visual overlays for pathologist review

## Data
N=30 bone marrow biopsies (15 CHIP, 15 non-CHIP) with αSMA, Masson's Trichrome, and Reticulin staining.

## Requirements
See `requirements.txt`.

## Usage
```bash
pip install -e .
python scripts/preprocess.py
python scripts/evaluate.py
```


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