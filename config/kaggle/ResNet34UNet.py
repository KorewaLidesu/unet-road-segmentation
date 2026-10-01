"""GohVh's ResNet34-UNet on the DeepGlobe road dataset: 300 epochs, sized to be fast on Kaggle.

Every knob reads an environment variable first, so the notebook can retune the
run without editing this file:

    CROP_SIZE=384 MAX_EPOCH=300 python train_supervision.py -c config/kaggle/ResNet34UNet.py

The network is ``UnetResnet34`` from GohVh/resnet34-unet, imported unmodified
from the pinned submodule (see ``geoseg/models/resnet34_unet.py``). So is the
optimisation recipe of upstream's ``main.py``: AdamW at lr 1e-3, weight decay
1e-4, under OneCycleLR stepped every batch. What differs, and why:

* **An epoch is one random CROP_SIZE window from each training scene** (default
  256), not a pass over every 512 tile. 300 passes over all ~20k tiles is ~6M
  512x512 training images, over a day of GPU time even on a T4 x2; this way every
  scene is still seen each epoch and the window lands somewhere new every time.
* **Cross-entropy + Dice** instead of plain cross-entropy: roads are ~4% of the
  pixels, and Dice stops the thin class drowning in background. It is the same
  CE + Dice the RFM-UNet runs trained with. ``LOSS=ce`` gives upstream's loss.
* Two classes (road, background) instead of upstream's 24 drone classes, and
  no early stopping -- all 300 epochs run and the best checkpoint is kept.
* For speed: 16-bit mixed precision, channels_last on tensor-core GPUs, cuDNN
  benchmark mode, batch 16 per GPU with no accumulation, and validation every
  5 epochs. A wall-clock stop leaves a resumable checkpoint before Kaggle's 12h
  limit.
"""

import os

import torch
from torch.utils.data import DataLoader

from geoseg.datasets.dpgb import CLASSES, DpgbDataset, DpgbSceneCropDataset, get_validation_transform
from geoseg.losses import DiceLoss, JointLoss, SoftCrossEntropyLoss
from geoseg.models.resnet34_unet import ResNet34UNet


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    if cast is bool:
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    return cast(raw)


def _devices(raw):
    """Trainer(devices=...) wants an int, a list, or the string 'auto'."""
    if raw == 'auto':
        return 'auto'
    if ',' in raw:
        return [int(x) for x in raw.split(',') if x.strip() != '']
    return int(raw)


# -------------------------------------------------------------------- paths --
# Set by tools/prepare_deepglobe.py, or pointed at an already-prepared Kaggle Dataset.
data_root = _env('DEEPGLOBE_ROOT', '/kaggle/working/data/deepglobe')
out_root = _env('OUT_ROOT', '/kaggle/working')

model_name = 'ResNet34UNet'
dataset_name = 'deepglobe'
weights_name = _env('WEIGHTS_NAME', model_name)
weights_path = _env('WEIGHTS_PATH', f'{out_root}/model_weights/{dataset_name}/{weights_name}')
test_weights_name = weights_name
# train_supervision keys its per-dataset metric averaging off this string. 'dpgb'
# is not in FOREGROUND_ONLY_DATASETS, so mIoU averages both classes, as in the
# RFM-UNet runs; read *_IoU_Road for the road-only number.
log_name = f'{weights_path}/lightning_logs/dpgb_log'

# ------------------------------------------------------------------ mlflow --
experiment_name = _env('MLFLOW_EXPERIMENT_NAME', 'ResNet34UNet-DeepGlobe')
run_name = _env('MLFLOW_RUN_NAME', f'{weights_name}-{dataset_name}')
# Only used when no MLFLOW_TRACKING_URI is configured (env var or Kaggle Secret).
mlflow_default_dir = _env('MLFLOW_DEFAULT_DIR', f'{out_root}/mlruns')
mlflow_log_model = _env('MLFLOW_LOG_MODEL', False, bool)  # checkpoints are ~450MB with AdamW state
mlflow_system_metrics = _env('MLFLOW_SYSTEM_METRICS', True, bool)
log_prediction_images = _env('LOG_PREDICTION_IMAGES', True, bool)
prediction_image_samples = _env('PREDICTION_IMAGE_SAMPLES', 4, int)
# Previews are drawn on validation epochs, so keep this a multiple of CHECK_VAL_EVERY_N_EPOCH.
prediction_image_every_n_epochs = _env('PREDICTION_IMAGE_EVERY_N_EPOCHS', 25, int)

# ---------------------------------------------------------- training hparam --
max_epoch = _env('MAX_EPOCH', 300, int)
# Training window per scene and epoch; a multiple of 64, at most the 512 tile.
# 256 -> 384 -> 512 is roughly 1x -> 2.25x -> 4x the training time.
crop_size = _env('CROP_SIZE', 256, int)
train_batch_size = _env('TRAIN_BATCH_SIZE', 16, int)  # per GPU
val_batch_size = _env('VAL_BATCH_SIZE', 16, int)
accumulate_grad_batches = _env('ACCUM_GRAD_BATCHES', 1, int)
lr = _env('LR', 1e-3, float)
weight_decay = _env('WEIGHT_DECAY', 1e-4, float)
loss_name = _env('LOSS', 'ce_dice')
num_classes = len(CLASSES)
classes = CLASSES
seed = _env('SEED', 42, int)
# cuDNN benchmark mode picks the fastest kernels per input shape; DETERMINISTIC=1
# trades a little speed for repeatable runs.
deterministic = _env('DETERMINISTIC', False, bool)

monitor = _env('MONITOR', 'val_mIoU')
monitor_mode = _env('MONITOR_MODE', 'max')
save_top_k = _env('SAVE_TOP_K', 1, int)
save_last = True
# Validation scores ~2.5k 512 tiles, which takes over half as long as a training
# epoch at CROP_SIZE=256; every epoch would make the run ~1.5x longer.
# Checkpoints are written after each validation.
check_val_every_n_epoch = _env('CHECK_VAL_EVERY_N_EPOCH', 5, int)
# Otherwise the last epochs -- under OneCycle, the lowest-LR and usually best
# ones -- are never validated or saved.
assert max_epoch % check_val_every_n_epoch == 0, (
    f'MAX_EPOCH={max_epoch} must be a multiple of CHECK_VAL_EVERY_N_EPOCH={check_val_every_n_epoch}')
pretrained_ckpt_path = _env('PRETRAINED_CKPT_PATH', None)

# ------------------------------------------------------------ kaggle runtime --
gpus = _devices(_env('GPUS', '1'))
strategy = _env('STRATEGY', 'auto')  # 'ddp' for T4 x2 (works because training runs as a subprocess)
precision = _env('PRECISION', '16-mixed')
# NHWC tensors run faster on tensor cores (compute >= 7.0); 'auto' leaves a P100 on NCHW.
channels_last = _env('CHANNELS_LAST', 'auto')
gradient_clip_val = _env('GRADIENT_CLIP_VAL', 0.0, float) or None  # upstream does not clip
num_workers = _env('NUM_WORKERS', 2, int)  # per GPU
log_every_n_steps = _env('LOG_EVERY_N_STEPS', 50, int)
# Steps between progress-bar redraws; 0 hides the bar. A redraw per step floods a
# notebook log over 300 epochs.
progress_bar_refresh_rate = _env('PROGRESS_BAR_REFRESH_RATE', 25, int)
num_sanity_val_steps = _env('NUM_SANITY_VAL_STEPS', 2, int)
# DD:HH:MM:SS. Kaggle cuts GPU sessions at 12h; stopping earlier leaves room for
# the final checkpoint, evaluation and saving the output. MAX_TIME= (empty)
# disables it, which _env() would read as "unset".
max_time = os.environ.get('MAX_TIME', '00:10:30:00') or None
auto_resume = _env('AUTO_RESUME', True, bool)
resume_ckpt_path = _env('RESUME_CKPT_PATH', None)
# Fractions < 1.0 (or batch counts > 1) make a smoke test out of this config.
limit_train_batches = _env('LIMIT_TRAIN_BATCHES', 1.0, float)
limit_val_batches = _env('LIMIT_VAL_BATCHES', 1.0, float)
# clDice/APLS skeletonise and graph every image on the CPU; keep them for the
# test script unless you are ready to pay for them each epoch.
train_topology_metrics = _env('TRAIN_TOPOLOGY_METRICS', False, bool)
val_topology_metrics = _env('VAL_TOPOLOGY_METRICS', False, bool)

# --------------------------------------------------------------- the network --
net = ResNet34UNet(num_classes=num_classes)

# ----------------------------------------------------------------- the loss --
if loss_name == 'ce':
    loss = torch.nn.CrossEntropyLoss()  # upstream's
elif loss_name == 'ce_dice':
    # Prepared masks are {0, 1}, so ignore_index=255 never fires.
    loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=255),
                     DiceLoss(smooth=0.05, ignore_index=255), 1.0, 1.0)
else:
    raise ValueError(f"LOSS={loss_name!r}; expected 'ce_dice' or 'ce'")

# ----------------------------------------------------------- the dataloader --
assert crop_size % 64 == 0, f'CROP_SIZE={crop_size} must be a multiple of 64 for UnetResnet34'
train_dataset = DpgbSceneCropDataset(data_root=data_root, img_dir='train_images',
                                     mask_dir='train_masks', crop_size=crop_size)
val_dataset = DpgbDataset(data_root=data_root, mode='val', img_dir='val_images', mask_dir='val_masks',
                          transform=get_validation_transform())
test_dataset = DpgbDataset(data_root=data_root, mode='test', img_dir='test_images', mask_dir='test_masks',
                           transform=get_validation_transform())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=num_workers,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=num_workers > 0)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=num_workers > 0)

# ------------------------------------------------------------ the optimiser --
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)


def lr_scheduler(optimizer, total_steps):
    """Upstream's OneCycleLR. train_supervision builds it once the trainer knows
    how many optimiser steps the run takes on this many GPUs."""
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)


lr_scheduler_interval = 'step'
