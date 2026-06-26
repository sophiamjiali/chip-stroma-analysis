# ==============================================================================
# Script:           callbacks.py
# Purpose:          Defines callbacks for model training and sweeps.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import logging
import optuna
import torch

import lightning.pytorch as pl

from typing import Optional
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


def configure_callbacks(trial: Optional[optuna.trial.Trial] = None,
                        early_stopping_patience: int = 20,
                        early_stopping_min_delta: float = 1e-5):
    """
    Configures and returns all comebacks needed for model training. Note that 
    checkpoints are not saved.

    Parameters
    ----------
    trial                    : Optuna Trial, if running a sweep
    early_stopping_patience  : Epochs without val_loss improve. before stopping.
    early_stopping_min_delta : Minimum improvement threshold for early stopping.
    """

    logger.info("=" * 50)
    logger.info("Step 04: Callback Configuration")
    logger.info(f"- Early Stopping Patience: {early_stopping_patience}")
    logger.info(f"- Early Stopping Minimum Delta: {early_stopping_min_delta}")
    logger.info("-" * 50)

    logger.info("Successfully configured the LearningRateMonitor callback")
    logger.info("Successfully configured the GradientNormCallback callback")
    lr_monitor = LearningRateMonitor(logging_interval = "step")
    grad_norm_callback = GradientNormCallback(log_every_n_steps = 50)

    logger.info("Successfully configured the EarlyStopping callback")
    early_stop_callback = EarlyStopping(
        monitor                  = "val_loss",
        mode                     = "min",
        patience                 = early_stopping_patience,
        min_delta                = early_stopping_min_delta,
        strict                   = False,
        check_on_train_epoch_end = False,
        check_finite             = True
    )

    # Safe-guard against false initialization typing
    early_stop_callback.best_score = torch.tensor(float('inf'))

    callbacks = [
        early_stop_callback,
        lr_monitor,
        grad_norm_callback
    ]

    # Add the pruning callback if the instance is a sweep
    if trial is not None:
        logger.info("Sweep detected; Successfully configured "
                    "PyTorchLightningPruningCallback")
        callbacks.append(optuna.integration.PyTorchLightningPruningCallback(
            trial   = trial,
            monitor = "val_loss"
        ))

    logger.info("=" * 50)

    return callbacks


# ------------------------------------------------------------------------------

class GradientNormCallback(pl.Callback):
    """
    Logs the total L2 gradient norm after each optimiser step.

    Encoder collapse in high-dimensional methylation VAEs manifests as
    vanishing encoder gradients that are invisible in standard loss logs.
    Logging every N steps to limit overhead.

    Parameters
    ----------
    log_every_n_steps : Frequency of gradient norm logging. Default 50.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        total_norm = sum(
            p.grad.detach().data.norm(2).item() ** 2
            for p in pl_module.parameters()
            if p.grad is not None
        ) ** 0.5
        pl_module.log("grad_norm", total_norm)


# [END]