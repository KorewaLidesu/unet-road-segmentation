"""Lightning callbacks used by the Kaggle training run."""

from __future__ import annotations

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
        imgs, masks = batch["img"], batch["gt_semantic_seg"]
        preds = pl_module(imgs).argmax(dim=1)
        for i in range(min(self.num_samples, imgs.shape[0])):
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
