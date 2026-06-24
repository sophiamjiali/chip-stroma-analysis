# ==============================================================================
# Script:           datamodule.py
# Purpose:          PyTorch Datamodule for patches and masks
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/22/2026
# ==============================================================================

import pandas as pd
import lightning.pytorch as pl

from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from chip_stroma.data.dataset import VesselPatchDataset
from chip_stroma.segmentation.sampler import PositiveWeightedSampler
from chip_stroma.data.transforms import get_train_transforms, get_val_transforms


class VesselSegmentationDataModule(pl.LightningDataModule):
    """
    Patient-level k-fold DataModule for vessel segmentation.

    Each fold is an independent experimental unit — single fold evaluation
    driven externally by scripts/train.py to preserve per-fold variance
    and sweep compatibility (Varoquaux & Cheplygina, NeuroImage, 2022;
    Bradshaw et al., Radiol Artif Intell, 2023; Bouthillier et al.,
    NeurIPS, 2021).

    Training split uses PositiveWeightedSampler with per-patient inverse-
    frequency weighting to prevent high-density patient dominance
    (Buda et al., Neural Networks, 2018; Nair et al., PMC, 2020).

    Validation split uses identity transforms and sequential sampling to
    produce unbiased performance estimates (Varoquaux & Cheplygina,
    NeuroImage, 2022).

    Parameters
    ----------
    manifest : pd.DataFrame
        Pre-filtered patch manifest (include=True rows only) with
        'fold', 'has_vessel', 'sample_id', 'patch_name', 'vessel_mask',
        'tissue_mask', 'patient_positive_count' populated.
    patch_dir : Path
        Directory of Vahadane-normalized patch PNGs.
    vessel_mask_dir : Path
        Directory of binary vessel annotation masks (0/255, mode L).
    tissue_mask_dir : Path
        Directory of binary tissue masks (0/255, mode L).
    fold : int
        Validation fold index; all other folds used for training.
    batch_size : int
        Number of patches per batch.
    num_workers : int
        DataLoader worker processes.
    pos_prob : float
        Probability of sampling a positive patch per draw (Buda et al.,
        Neural Networks, 2018). Default: 0.5.
    seed : int
        Random seed for sampler reproducibility (Colliot et al.,
        ML for Brain Disorders, 2023).
    """

    def __init__(self,
                 manifest:        pd.DataFrame,
                 patch_dir:       Path,
                 vessel_mask_dir: Path,
                 tissue_mask_dir: Path,
                 fold:            int,
                 batch_size:      int,
                 num_workers:     int,
                 pos_prob:        float = 0.5,
                 seed:            int = 42) -> None:
        
        super().__init__()

        self.manifest        = manifest
        self.patch_dir       = patch_dir
        self.vessel_mask_dir = vessel_mask_dir
        self.tissue_mask_dir = tissue_mask_dir
        self.fold            = fold
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.pos_prob        = pos_prob
        self.seed            = seed

        self.train_dataset: Optional[VesselPatchDataset] = None
        self.val_dataset:   Optional[VesselPatchDataset] = None


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Partition manifest into train/val splits by fold index.

        Called by Lightning before fit(); split performed here rather
        than __init__ to support multi-GPU DataModule replication
        (PyTorch Lightning docs, 2024).
        """

        # Extract the train and validation sets based on the current fold
        train_manifest = (self.manifest[self.manifest['fold'] != self.fold]
                          .reset_index(drop = True))
        val_manifest   = (self.manifest[self.manifest['fold'] == self.fold]
                          .reset_index(drop = True))
        
        # Initialize the train dataset with full transform protocol
        self.train_dataset = VesselPatchDataset(
            manifest        = train_manifest,
            patch_dir       = self.patch_dir,
            vessel_mask_dir = self.vessel_mask_dir,
            tissue_mask_dir = self.tissue_mask_dir,
            transform       = get_train_transforms()
        )

        # Initialize the validation dataset with no transforms
        self.val_dataset = VesselPatchDataset(
            manifest        = val_manifest,
            patch_dir       = self.patch_dir,
            vessel_mask_dir = self.vessel_mask_dir,
            tissue_mask_dir = self.tissue_mask_dir,
            transform       = get_val_transforms()
        )

    # =====| DataLoaders |======================================================

    def train_dataloader(self) -> DataLoader:
        """
        Training DataLoader with PositiveWeightedSampler.

        shuffle = False: ordering delegated to sampler, which interleaves
        positives and negatives to prevent batch normalization drift
        (Ioffe & Szegedy, ICML, 2015).
        drop_last = True: prevents batch norm instability from undersized
        final batches (Ioffe & Szegedy, ICML, 2015).
        pin_memory = True: accelerates CPU-to-GPU transfer for patch tensors
        (PyTorch docs, 2024).
        """

        assert self.train_dataset is not None, (
            "train_dataset is None - ensure setup()"
            " is called before train_dataloader()"
        )

        sampler = PositiveWeightedSampler(
            manifest = self.train_dataset.records,
            pos_prob = self.pos_prob,
            seed     = self.seed
        )

        return DataLoader(
            self.train_dataset,
            batch_size         = self.batch_size,
            sampler            = sampler,
            num_workers        = self.num_workers,
            pin_memory         = False,
            persistent_workers = True,
            drop_last          = True
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Validation DataLoader with sequential sampling.

        shuffle = False: validation must be deterministic for reproducible
        metric estimation (Varoquaux & Cheplygina, NeuroImage, 2022).
        drop_last = False: all validation patches must be evaluated to
        avoid biased Dice estimation at small fold sizes (Bradshaw et al.,
        Radiol Artif Intell, 2023).
        """

        assert self.val_dataset is not None, (
            "val_dataset is None - ensure setup()"
            " is called before val_dataloader()"
        )

        return DataLoader(
            self.val_dataset,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = False,
            persistent_workers = True,
            drop_last          = False
        )
    
# [END]