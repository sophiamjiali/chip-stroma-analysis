# ==============================================================================
# Script:           callbacks.py
# Purpose:          Defines callbacks for model training and sweeps.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/24/2026
# ==============================================================================

import shutil
import optuna
import json
import wandb

import lightning.pytorch as pl

from pathlib import Path
from box import Box
from typing import Optional
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from chip_stroma.utils.loggers import setup_logger

logger = setup_logger(__name__)


def configure_callbacks(params: Box,
                        trial: Optional[optuna.trial.Trial] = None):
                        
    """
    Configures and returns all comebacks needed for model training. Note that 
    checkpoints are not saved.

    Parameters
    ----------
    trial                    : Optuna Trial, if running a sweep
    early_stopping_patience  : Epochs without val/loss improve. before stopping.
    early_stopping_min_delta : Minimum improvement threshold for early stopping.
    """

    logger.info("=" * 50)
    logger.info("Step 04: Callback Configuration")
    logger.info(f"- Early Stopping Patience: {params.early_stopping.patience}")
    logger.info(f"- Early Stopping Minimum Delta: {params.early_stopping.min_delta}")
    logger.info("-" * 50)

    logger.info("Successfully configured the LearningRateMonitor callback")
    logger.info("Successfully configured the GradientNormCallback callback")
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")
    grad_norm_callback = GradientNormCallback(log_every_n_steps = 50)

    logger.info("Successfully configured the EarlyStopping callback")
    early_stop_callback = EarlyStopping(
        monitor                  = str(params.early_stopping.metric),
        mode                     = str(params.early_stopping.mode),
        patience                 = params.early_stopping.patience,
        min_delta                = params.early_stopping.min_delta,
        strict                   = False,
        check_on_train_epoch_end = False,
        check_finite             = False,
        verbose                  = True
    )

    logger.info("Successfulyl configured the BestEpochTracker callback")
    tracker = BestEpochTracker(
        monitor = str(params.epoch_tracker.metric),
        mode    = str(params.epoch_tracker.mode)
    )

    callbacks = [
        early_stop_callback,
        lr_monitor,
        grad_norm_callback,
        tracker
    ]

    # Add the pruning callback if the instance is a sweep
    if trial is not None:
        logger.info("Sweep detected; Successfully configured "
                    "PyTorchLightningPruningCallback")
        callbacks.append(optuna.integration.PyTorchLightningPruningCallback(
            trial   = trial,
            monitor = params.pruner.metric
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


class BestEpochTracker(pl.Callback):
    """Tracks the best epoch by a monitored metric."""

    def __init__(self, monitor: str = "val/dice", mode: str = "max"):
        super().__init__()
        assert mode in ("max", "min")
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Pull the monitored metric from Lightning's logged metrics
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return  # metric not yet logged (e.g. sanity check epoch)

        current = metrics[self.monitor].item()
        improved = (current > self.best_score) if self.mode == "max" else (current < self.best_score)

        if improved:
            self.best_score = current
            self.best_epoch = trainer.current_epoch

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Write final best epoch/score to W&B summary (not per-step log)
        if wandb.run is not None:
            wandb.run.summary[f"best_{self.monitor.replace('/', '_')}"] = self.best_score
            wandb.run.summary["best_epoch"] = self.best_epoch


# =====| Checkpointing |========================================================

def make_checkpoint_callback(checkpoint_dir: Path,
                             group: str):
    """Wrapper for study-level callback. Keeps only one checkpoint 
    corresponding to the best (current) trial of the sweep."""

    def save_best_after_trial(study: optuna.Study, 
                              trial: optuna.trial.FrozenTrial) -> None:
        """Optuna callback — fires after each trial completes."""

        # Skip pruned or failed trials
        if trial.state != optuna.trial.TrialState.COMPLETE: return

        try: best_trial = study.best_trial
        except ValueError: return

        trial_ckpt = checkpoint_dir / Path(group) / f"trial_{trial.number}.ckpt"

        # If the best trial was found, delete all other checkpoint(s)
        if best_trial.number == trial.number:
            best_ckpt = checkpoint_dir / Path(group) / "best_trial.ckpt"
            json_path = checkpoint_dir / Path(group) / "best_trial.json"
            shutil.copy(trial_ckpt, best_ckpt)

            with open(json_path, "w") as f:
                json.dump({
                    "trial_number": trial.number, 
                    "params"      : trial.params, 
                    "val_dice"    : trial.value,
                    "encoder_name": "resnet34"
                }, f)

        # Delete this trials checkpoint; turned into best-so-far if so
        trial_ckpt.unlink(missing_ok = True)
                
    return save_best_after_trial


# [END]