"""RFM-UNet on the DeepGlobe road dataset, sized for a Kaggle GPU session.

Every knob reads an environment variable first, so the notebook can retune the
run without editing this file:

    TRAIN_BATCH_SIZE=4 MAX_EPOCH=40 python train_supervision.py -c config/kaggle/RFMUNet.py

Differences from upstream ``config/DPGB/RFMUNet.py`` and why:

* paths point at ``/kaggle/{input,working}`` instead of ``/root/autodl-tmp``;
* batch 2 x 6 accumulation steps instead of batch 12, because 512x512 RFM-UNet
  activations do not fit in a 16GB T4 at batch 12 -- the effective batch, and so
  the optimiser's view of the run, is unchanged;
* ``use_checkpoint=True`` on the Mamba encoder, the other half of that trade;
* 16-bit mixed precision;
* a wall-clock stop so the session saves a resumable checkpoint before Kaggle
  pulls the plug.

The learning rate, optimiser, schedule, loss and augmentations are upstream's.
"""

import os

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.dpgb import *
from tools.utils import Lookahead
from tools.utils import process_model_params


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
# Set by tools/prepare_deepglobe.py, or pointed at an already-prepared Kaggle
# Dataset. geoseg.datasets.dpgb reads the same variable for its module-level
# `path`, which is what the dataset constructors below default to.
data_root = _env('DEEPGLOBE_ROOT', '/kaggle/working/data/deepglobe')
path = data_root
out_root = _env('OUT_ROOT', '/kaggle/working')

dataset_name = 'deepglobe'
weights_name = _env('WEIGHTS_NAME', 'RFMUNet')
weights_path = _env('WEIGHTS_PATH', f'{out_root}/model_weights/{dataset_name}/{weights_name}')
test_weights_name = weights_name
# Kept because train_supervision keys its per-dataset metric averaging off this
# string. 'dpgb' is not in FOREGROUND_ONLY_DATASETS, so mIoU averages both
# classes exactly as upstream's DeepGlobe config does; read *_IoU_Road for the
# road-only number.
log_name = f'{weights_path}/lightning_logs/dpgb_log'

# ------------------------------------------------------------------ mlflow --
experiment_name = _env('MLFLOW_EXPERIMENT_NAME', 'RFMUNet-DeepGlobe')
run_name = _env('MLFLOW_RUN_NAME', f'{weights_name}-{dataset_name}')
# Only used when no MLFLOW_TRACKING_URI is configured (env var or Kaggle Secret).
mlflow_default_dir = _env('MLFLOW_DEFAULT_DIR', f'{out_root}/mlruns')
mlflow_log_model = _env('MLFLOW_LOG_MODEL', False, bool)  # checkpoints are ~300MB
mlflow_system_metrics = _env('MLFLOW_SYSTEM_METRICS', True, bool)
log_prediction_images = _env('LOG_PREDICTION_IMAGES', True, bool)
prediction_image_samples = _env('PREDICTION_IMAGE_SAMPLES', 4, int)
prediction_image_every_n_epochs = _env('PREDICTION_IMAGE_EVERY_N_EPOCHS', 5, int)

# ---------------------------------------------------------- training hparam --
max_epoch = _env('MAX_EPOCH', 105, int)
ignore_index = len(CLASSES)
train_batch_size = _env('TRAIN_BATCH_SIZE', 2, int)
val_batch_size = _env('VAL_BATCH_SIZE', 2, int)
accumulate_grad_batches = _env('ACCUM_GRAD_BATCHES', 6, int)  # 2 x 6 = upstream's 12
lr = _env('LR', 1e-3, float)
weight_decay = _env('WEIGHT_DECAY', 0.0025, float)
backbone_lr = _env('BACKBONE_LR', 1e-3, float)
backbone_weight_decay = _env('BACKBONE_WEIGHT_DECAY', 0.0025, float)
num_classes = len(CLASSES)
classes = CLASSES
seed = _env('SEED', 42, int)

monitor = _env('MONITOR', 'val_mIoU')
monitor_mode = _env('MONITOR_MODE', 'max')
save_top_k = _env('SAVE_TOP_K', 1, int)
save_last = True
check_val_every_n_epoch = _env('CHECK_VAL_EVERY_N_EPOCH', 1, int)
pretrained_ckpt_path = _env('PRETRAINED_CKPT_PATH', None)

# ------------------------------------------------------------ kaggle runtime --
gpus = _devices(_env('GPUS', '1'))
strategy = _env('STRATEGY', 'auto')  # 'ddp' for T4 x2 (works because training runs as a subprocess)
precision = _env('PRECISION', '16-mixed')
# Upstream does not clip. Set GRADIENT_CLIP_VAL=1.0 if fp16 sends the loss to NaN.
gradient_clip_val = _env('GRADIENT_CLIP_VAL', 0.0, float) or None
num_workers = _env('NUM_WORKERS', 2, int)
log_every_n_steps = _env('LOG_EVERY_N_STEPS', 50, int)
num_sanity_val_steps = _env('NUM_SANITY_VAL_STEPS', 2, int)
# DD:HH:MM:SS. Kaggle cuts GPU sessions at 12h; stopping earlier leaves room for
# the final checkpoint write. Set to '' to disable.
max_time = _env('MAX_TIME', '00:08:00:00') or None
auto_resume = _env('AUTO_RESUME', True, bool)
resume_ckpt_path = _env('RESUME_CKPT_PATH', None)
# Fractions < 1.0 make a smoke test out of this config.
limit_train_batches = _env('LIMIT_TRAIN_BATCHES', 1.0, float)
limit_val_batches = _env('LIMIT_VAL_BATCHES', 1.0, float)
# clDice/APLS skeletonise and graph every image on the CPU; keep them for the
# test script unless you are ready to pay for them each epoch.
train_topology_metrics = _env('TRAIN_TOPOLOGY_METRICS', False, bool)
val_topology_metrics = _env('VAL_TOPOLOGY_METRICS', False, bool)

# --------------------------------------------------------------- the network --
from geoseg.models.RFMUNet import RFMUNet

use_checkpoint = _env('USE_GRAD_CHECKPOINT', True, bool)
# VMamba-tiny ImageNet weights for the encoder, loaded non-strictly (the ADAMamba
# blocks add parameters VMamba has none of). Off by default to match upstream,
# but worth trying: DeepGlobe is small for training an SSM encoder from scratch.
pretrained_encoder = _env('PRETRAINED_ENCODER', False, bool)
net = RFMUNet(num_classes=num_classes, pretrained=pretrained_encoder,
              use_checkpoint=use_checkpoint)

# ----------------------------------------------------------------- the loss --
loss = EdgeLoss(ignore_index=255)  # upstream's value; prepared masks are {0,1} so nothing is ignored
use_aux_loss = False

# ----------------------------------------------------------- the dataloader --
train_dataset = DpgbDataset(data_root=path, mode='train', img_dir='train_images', mask_dir='train_masks', mosaic_ratio=_env('MOSAIC_RATIO', 0.25, float), transform=get_training_transform())
val_dataset = DpgbDataset(data_root=path, mode='val', img_dir='val_images', mask_dir='val_masks', transform=get_validation_transform())
test_dataset = DpgbDataset(data_root=path, mode='test', img_dir='test_images', mask_dir='test_masks', transform=get_validation_transform())

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

######################## optimizer_config ######################
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
