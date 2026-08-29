# chip-stroma-analysis

Computational pathology pipeline for quantifying αSMA⁺ stromal (fibroblast/vessel) content in bone marrow biopsies, and testing whether Clonal Hematopoesis of Indeterminate Potential (CHIP) patients show elevated stromal remodeling relative to non-CHIP patients.

## Table of Contents
- [Motivation](#motivation)
- [Study Design](#study-design)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data](#data)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Known Limitations & Open Items](#known-limitations--open-items)
- [Roadmap](#roadmap)
- [Citing This Work](#citing-this-work)
- [License](#license)
- [Contact](#contact)

## Motivation
Prior wet-lab work in this group observed that CHIP patients exhibit increased αSMA⁺ fibroblast/vessel content in bone marrow biopsies compared to non-CHIP patients, consistent with a pro-fibrotic stromal remodeling phenotype. This repository implements a **computational pathology pipeline** to replicate that finding at scale from whole-slide IHC images, using a deep learning vessel segmentation model followed by stain-deconvolution-based fibroblast quantification and patient-level statistical testing.
 
The primary scientific endpoint is an **unpaired t-test** comparing CHIP vs. non-CHIP patients on αSMA⁺ object density, matched as closely as possible to the original wet-lab quantification protocol.

## Study Design
 
| | |
|---|---|
| **Modality** | αSMA H-DAB immunohistochemistry, whole-slide images |
| **Patch extraction** | 256×256 px, via [TRIDENT](https://github.com/mahmoodlab/TRIDENT) |
| **Cohort size** | ~24–30 patients (~15 CHIP, ~15 non-CHIP) |
| **Class imbalance** | ~125:1 background : vessel pixels |
| **Primary endpoint** | `object_density` (discrete αSMA⁺ object count / tissue area), unpaired Welch's t-test |
| **Secondary endpoint** | `fibroblast_density` (αSMA⁺ pixel-area fraction) |
| **Cross-validation** | Patient-level 5-fold `StratifiedGroupKFold` |
| **Known outlier** | `h-BMO-18` — contributes ~21.6% of all filtered positive patches |

## Repository Structure
 
```
chip-stroma-analysis/
├── configs/                     # YAML configs, one per pipeline stage
│   ├── 00_paths.yaml
│   ├── 01_preprocess.yaml 
│   ├──  ...
│   ├── 12_stitch_masks.yaml
│   ├── pipeline.yaml             # stage name -> step number mapping
│   └── sweeps/                   # Optuna HPO sweep configs (v1–v8)
├── scripts/                     # Thin orchestration entrypoints (01–12)
├── slurm/                       # SLURM batch/array job wrappers per stage
└── src/chip_stroma/
│   ├── data/                     # Preprocessing, Dataset, DataModule, sampler, transforms
│   ├── models/                   # U-Net (SMP), Lightning module, losses, inference
│   ├── training/                 # train/objective/CV/Optuna study management
│   ├── quantification/           # Fibroblast density scoring
│   ├── classification/           # Statistical replication (t-test, LOOCV AUC)
│   ├── evaluate/                 # Segmentation performance statistics
│   ├── visualize/                # Figure generation, WSI overlay stitching
│   └── utils/                    # Config, I/O, logging, callbacks, misc.
```

## Pipeline Overview
| Step | Script | Purpose |
|---|---|---|
| 00 | `00_setup_env.sh` | Environment activation |
| 01 | `01_preprocess.py` | Tissue/artifact filtering, Vahadane normalization, vessel mask conversion |
| 02 | `02_assign_folds.py` | CHIP label assignment, patient-stratified 5-fold CV assignment |
| 03 | `03_train.py` | Single-configuration U-Net training |
| 04 | `04a_init_study.py`, `04b_sweep.py` | Optuna HPO sweep (fold 0, TPE sampler, median pruning) |
| 05 | `05a_multiseed.py`, `05b_aggregate_multiseed.py` | Multi-seed confirmation of top-K sweep trials |
| 06 | `06a_full_cv.py`, `06b_aggregate_full_cv.py` | Full 5-fold CV retrain with frozen hyperparameters |
| 07 | `07_inference.py` | Out-of-fold inference → quantized probability maps (`val_arrays.h5`) |
| 08 | `08_evaluate.py` | Segmentation metrics, threshold sweep, calibration, Optuna diagnostics |
| 09 | `09_visualize.py` | QC figure generation |
| 10 | `10_quantify.py` | αSMA⁺ fibroblast density + object count quantification |
| 11 | `11_analysis.py` | CHIP vs. non-CHIP statistical testing |
| 12 | `12_stitch_masks.py` | WSI-level mask stitching (vessel / fibroblast / tissue), QuPath GeoJSON export |
 
Each stage is submitted independently via `slurm/submit_job.sh <step_name> <version>`,
which resolves the step number from `configs/pipeline.yaml` and dispatches the
corresponding SLURM script.

## Installation
 
```bash
git clone <repo-url>
cd chip-stroma-analysis
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
 
Cluster-side (SLURM/HPC), the environment is instead a pre-built venv sourced
directly in `slurm/00_setup_env.sh`:
 
```bash
source /cluster/home/<user>/envs/chip-stroma-env-gpu/bin/activate
```
 
**Key dependencies:** PyTorch Lightning, `segmentation-models-pytorch`,
Optuna, Weights & Biases, `tiatoolbox`, `scikit-image`, Albumentations,
`h5py`. See `requirements.txt` / `pyproject.toml` for pinned versions.
 
> ⚠️ **Do not use MONAI** in this project — incompatible with current loss
> function / metric conventions used throughout `src/chip_stroma/models`.

## Configuration
 
Configuration is split into a **path config** (`configs/00_paths.yaml`) and
one **stage-specific config** per script (`configs/0N_<stage>.yaml`), merged
at runtime via `load_configs()` into a frozen/unfrozen `Box` object. Paths
resolve relative to `PROJECT_ROOT` set in `.env`.
 
```bash
python scripts/01_preprocess.py --config_dir configs/
```
 
HPO sweep configs live separately under `configs/sweeps/{version}.yaml`;
any parameter provided as a `[min, max]` list is swept by Optuna, any scalar
is held constant — enabling fast toggling between exploratory mini-sweeps and
full sweeps without code changes.
 
## Usage
 
Typical end-to-end run on the SLURM cluster:
 
```bash
# 1. Preprocess raw patches (tissue/artifact filtering, normalization)
./slurm/submit_job.sh preprocess v1
 
# 2. Assign CHIP labels + patient-stratified folds
./slurm/submit_job.sh fold_assignment v1
 
# 3. Hyperparameter sweep (Optuna, fold 0)
./slurm/04_run_sweep.sh v8
 
# 4. Multi-seed confirmation of top-K trials
./slurm/05_run_multiseed.sh v8
 
# 5. Full 5-fold CV retrain with frozen hyperparameters
./slurm/06_run_full_cv.sh v8
 
# 6. Out-of-fold inference
./slurm/submit_job.sh inference v8
 
# 7. Evaluation, quantification, statistical analysis
./slurm/submit_job.sh evaluate v8
./slurm/submit_job.sh quantify v8
./slurm/submit_job.sh analysis v8
```
 
## Data
 
- **Input:** αSMA H-DAB whole-slide IHC images, patch-extracted via TRIDENT.
- **Cohort:** ~24–30 bone marrow biopsies (~15 CHIP / ~15 non-CHIP),
  labeled via `CHIP_IDs.csv` matched by substring to sanitized sample IDs.
- **Raw data is not versioned in this repository** (`data/` is gitignored) —
  paths are resolved via `.env` + `configs/00_paths.yaml` to cluster-mounted
  storage (`/cluster/projects/kumargroup/...`).
- Manifests (`patch_manifest.csv`, `train_manifest.csv`) are the canonical
  join keys throughout the pipeline: **`sample_id` + `patch_name` together**
  (not `patch_name` alone, which collides across patients).
## Modeling Approach
 
- **Architecture:** U-Net (`segmentation_models_pytorch`), ResNet-34 encoder,
  ImageNet-pretrained.
- **Loss:** Focal Tversky Loss (Abraham & Khan, 2019) + Masked Dice, with a
  linearly-ramped Boundary Loss term (Kervadec et al., 2019), all restricted
  to tissue pixels and normalized by foreground pixel count to prevent
  background dominance under ~125:1 class imbalance.
- **Sampling:** Patient-balanced positive oversampling
  (`PositiveWeightedSampler`) — per-patch weights are inverse-frequency by
  patient to prevent `h-BMO-18` from dominating gradient signal.
- **Model selection vs. early stopping (decoupled):** `EarlyStopping` on
  `val/loss` (min, patience 25); best-epoch tracking on `val/dice` (max) —
  prevents premature stopping from an early Dice plateau.
- **HPO:** Optuna TPE sampler + median pruner, `JournalStorage` (NFS-safe;
  SQLite over NFS is unreliable for concurrent SLURM array workers).
- **CV protocol:** Hyperparameters are tuned only on fold 0, frozen after
  multi-seed confirmation, then applied unmodified across all 5 folds for
  unbiased performance estimates (Bradshaw et al., 2023; Cawley & Talbot,
  2010). A separate all-patient retrain (not yet implemented) is required
  for deployment.
- **Metric aggregation:** Per-patient **macro** averaging (nanmean within
  patient, then across patients) is the primary reported metric; global
  micro-averaged metrics are logged for diagnostics only.

## Reproducibility
 
- All random seeds are explicit and threaded through config
  (`pl.seed_everything`, sampler seed, Optuna sampler seed).
- `trainer.deterministic = True` for all Lightning trainers.
- Swept Optuna parameters are always written back into a `deepcopy`'d config
  object per trial (`trial_params = deepcopy(params)`) to prevent
  shared-object mutation bugs across trials.
- Empty-mask / no-signal patches are excluded from metric aggregation via a
  `has_signal` (GT-positivity-only) guard, following Reinke et al.
  (*Nature Methods*, 2023) — not a pred+GT union guard.
- Stain normalization: **Vahadane**, not Macenko — Macenko's H&E-specific
  SVD assumptions degrade on H-DAB IHC tissue.
- Stain deconvolution: `skimage.color.separate_stains` with `hdx_from_rgb`
  (2-stain IHC matrix; DAB at channel index 1) — not `rgb2hed`.
- Model checkpoints, OOF probability maps (quantized uint8, `val_arrays.h5`),
  and all statistical outputs are versioned by an explicit `--version` tag
  passed to every pipeline stage.

## Citing This Work
 
This project has not yet been published. If you use this pipeline, please
cite the repository directly and contact the author for the associated
manuscript status.
 
```bibtex
@software{li_chip_stroma_2026,
  author  = {Li, Sophia Mengjia},
  title   = {chip-stroma-analysis: Computational Pathology Pipeline for
             CHIP-Associated Stromal Remodeling},
  year    = {2026},
  note    = {CCG Lab, Princess Margaret Cancer Centre, UHN, University of Toronto}
}
```
 
## License
 
Not yet finalized for public release — internal CCG Lab / UHN research use.
Contact the author before reuse or redistribution.
 
## Contact
 
**Sophia Mengjia Li**
CCG Lab, Princess Margaret Cancer Centre, UHN, University of Toronto
sophiamjia.li@mail.utoronto.ca