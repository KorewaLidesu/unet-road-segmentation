"""Lightning callbacks used by the Kaggle training run."""

from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from tools import mlflow_utils

# albumentations' Normalize() defaults, needed to turn a batch back into pixels.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ROAD = 0  # class index of road, per dpgb.PALETTE ([255,255,255] first)


def denormalize(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) uint8."""
    arr = img.detach().float().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Road white, background black."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == ROAD] = 255
    return out


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Green = hit, red = false positive, blue = missed road."""
    gt_road, pred_road = gt == ROAD, pred == ROAD
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[gt_road & pred_road] = (0, 200, 0)
    out[~gt_road & pred_road] = (220, 0, 0)
    out[gt_road & ~pred_road] = (0, 80, 255)
    return out


def _hstack(panels: list[np.ndarray], pad: int = 4) -> np.ndarray:
    gap = np.full((panels[0].shape[0], pad, 3), 32, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        if i:
            stacked.append(gap)
        stacked.append(panel)
    return np.concatenate(stacked, axis=1)


def _hms(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


class EpochTimer(Callback):
    """Log each epoch's wall time as ``epoch_seconds`` and print an ETA.

    In Lightning 2.x ``on_train_epoch_end`` fires after that epoch's validation,
    so the time includes it, and the ETA -- the mean over the epochs this
    session has run -- spreads the validation epochs' extra cost evenly.
    """

    def __init__(self):
        super().__init__()
        self._fit_start = self._epoch_start = None
        self._first_epoch = 0

    def on_train_start(self, trainer, pl_module):
        self._fit_start = time.perf_counter()
        self._first_epoch = trainer.current_epoch  # nonzero when resumed

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is None or trainer.global_rank != 0:
            return
        now = time.perf_counter()
        seconds = now - self._epoch_start
        epoch = trainer.current_epoch + 1
        mean = (now - self._fit_start) / max(1, epoch - self._first_epoch)
        remaining = max(0, trainer.max_epochs - epoch) * mean
        if trainer.logger is not None:
            trainer.logger.log_metrics({"epoch_seconds": seconds}, step=trainer.global_step)
        print(f"[speed] epoch {epoch}/{trainer.max_epochs} took {seconds:.1f}s, "
              f"~{_hms(remaining)} to go", flush=True)


class MLflowPredictionImages(Callback):
    """Log a few validation predictions to MLflow every N epochs.

    Four panels per sample: input, ground truth, prediction, error map. Cheap
    (one already-computed batch, rank zero only) and the fastest way to see
    whether a road model is learning connectivity or just blobs.
    """

    def __init__(self, num_samples: int = 4, every_n_epochs: int = 5):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = max(1, every_n_epochs)
        self._rows: list[np.ndarray] = []

    def _active(self, trainer) -> bool:
        if trainer.sanity_checking or trainer.global_rank != 0:
            return False
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self._rows = []

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0 or not self._active(trainer):
            return
        n = min(self.num_samples, batch["img"].shape[0])
        imgs, masks = batch["img"][:n], batch["gt_semantic_seg"][:n]
        # Hooks run outside Lightning's autocast; without this the previews would
        # be a second, full-precision forward pass.
        with trainer.precision_plugin.forward_context():
            preds = pl_module(imgs).argmax(dim=1)
        for i in range(n):
            gt = masks[i].cpu().numpy().astype(np.int64)
            pred = preds[i].cpu().numpy().astype(np.int64)
            self._rows.append(_hstack([
                denormalize(imgs[i]),
                mask_to_rgb(gt),
                mask_to_rgb(pred),
                error_map(gt, pred),
            ]))

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._rows or not self._active(trainer):
            return
        gap = np.full((4, self._rows[0].shape[1], 3), 32, dtype=np.uint8)
        grid: list[np.ndarray] = []
        for i, row in enumerate(self._rows):
            if i:
                grid.append(gap)
            grid.append(row)
        image = np.concatenate(grid, axis=0)
        mlflow_utils.log_image(
            trainer.logger,
            image,
            f"val_predictions/epoch_{trainer.current_epoch:04d}.png",
        )
        self._rows = []
