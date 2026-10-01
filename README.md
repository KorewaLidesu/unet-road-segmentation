# ResNet34-UNet road extraction on Kaggle, tracked with MLflow

[GohVh/resnet34-unet](https://github.com/GohVh/resnet34-unet) — a U-Net decoder on an
ImageNet-pretrained ResNet34 encoder — trained for 300 epochs on the [DeepGlobe Road
Extraction dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
in a Kaggle GPU session, with every run recorded in MLflow.

The network is upstream's `UnetResnet34`, **imported unmodified** from a git submodule pinned
at `af46a13`, and so are its optimiser and schedule. This repo supplies the DeepGlobe data
pipeline, a training loop sized so that 300 epochs take hours rather than days, evaluation
with road-topology metrics, and the tracking.

Start with [`resnet34unet-kaggle.ipynb`](resnet34unet-kaggle.ipynb). The RFM-UNet pipeline
this replaced lives on the
[`rfmunet-kaggle`](https://github.com/KorewaLidesu/unet-road-segmentation/tree/rfmunet-kaggle)
branch; both use the same data split and metrics, so their test numbers compare directly.

---

## Quick start

On Kaggle:

1. Accelerator **GPU T4 x2** (or P100), **Internet on**.
2. Attach `balraj98/deepglobe-road-extraction-dataset`.
3. Run [`resnet34unet-kaggle.ipynb`](resnet34unet-kaggle.ipynb) top to bottom, or *Save & Run
   All*. It probes the GPUs, clones this repo with the model submodule, prepares the data,
   runs a 2-batch smoke test on the same GPUs as the real run, trains, and evaluates.

Anywhere else:

```bash
git clone --recurse-submodules -b resnet34unet-kaggle \
  https://github.com/KorewaLidesu/unet-road-segmentation.git
cd unet-road-segmentation                        # already cloned? git submodule update --init
pip install -r requirements-kaggle.txt           # plus torch and torchvision
python tools/prepare_deepglobe.py --src /path/to/deepglobe --dst ./data/deepglobe
DEEPGLOBE_ROOT=./data/deepglobe OUT_ROOT=. \
  python train_supervision.py -c config/kaggle/ResNet34UNet.py
DEEPGLOBE_ROOT=./data/deepglobe OUT_ROOT=. \
  python road_seg_test.py -c config/kaggle/ResNet34UNet.py -o ./results --rgb -t lr
```

---

## Training speed: why 300 epochs fit in a session

The obvious epoch — every 512 tile of the ~5,000 training scenes, ~20,000 tiles — makes 300
epochs about 6 million 512×512 training images: over a day of GPU time even on a T4 x2. Here
**an epoch is one random window from each training scene**, 256×256 by default (`CROP_SIZE`):

- every scene is still seen every epoch, at a new position each time;
- an epoch costs a sixteenth of a full-resolution pass;
- windows are cut at native resolution, so roads are as wide in training as at test time.

On top of that:

- **DDP over both T4s** — the notebook switches it on when it sees two GPUs.
- **16-bit mixed precision** on the T4's tensor cores, in training and in evaluation.
- **channels_last** — +27% training throughput at 256² on the GPU measured below.
  `CHANNELS_LAST=auto` enables it on compute ≥ 7.0 only; a P100 has no tensor cores.
- **cuDNN benchmark mode.** `DETERMINISTIC=1` swaps it for repeatable kernels, ~3% slower here.
- **Batch 16 per GPU** with no gradient accumulation or checkpointing; at 256² the model uses
  under 2 GB.
- **Validation every 5 epochs.** One pass over the ~2,500 validation tiles takes over half as
  long as a training epoch.
- **Metrics accumulate on the GPU** and reach the host once per epoch, instead of copying every
  predicted mask to numpy at every step.

Measured on an RTX 3050 Ti Laptop GPU with data loading restricted to 2 cores / 4 threads (a
Kaggle box has 4 vCPUs), then projected to DeepGlobe's ~5,000 training scenes and ~2,500
validation tiles:

| `CROP_SIZE` | training images/s | per epoch | 300 epochs + validation | T4 x2 (estimate) |
| --- | --- | --- | --- | --- |
| **256** (default) | 118 | ~42 s | ~4 h | ~2 h |
| 384 | 53 | ~94 s | ~8 h | ~4 h |
| 512 | 30 | ~166 s | ~14 h | ~7 h |

Those rates are within 5–10% of the GPU's bare training-step rate, so the data loader keeps up.
A T4 is roughly the same class of card, and DDP over two of them should roughly halve the time;
the T4 column is that estimate, not a measurement. The real figure is printed after every epoch
(`[speed] epoch N/300 took …s, ~… to go`) and logged as `epoch_seconds`. A bigger window gives
the network more road context per sample for proportionally more time.

---

## Data preparation

`tools/prepare_deepglobe.py` handles two things that are easy to get wrong.

**Only `train/` is labelled.** DeepGlobe's `valid/` and `test/` directories are the
unlabelled competition holdouts. The 6226 labelled pairs in `train/` are split 80/10/10
into train/val/test. The split is on *source image id, before tiling*, so crops of one
scene never end up in two different splits.

**Masks must be class indices, not 0/255.** `geoseg/datasets/dpgb.py` feeds the mask
straight to the loss, and both `dpgb.PALETTE` and `tools/metric.py` treat **class 0 as
road** (white first in the palette; `_skeletonize` uses `mask == 0`). So the script writes
road → 0, background → 1. Handing the raw 0/255 mask to training instead would be silently
catastrophic: the loss's `ignore_index=255` would discard every road pixel and the model
would happily converge to all-background with a fine-looking OA. The script verifies the
encoding after writing rather than trusting it.

Source images are 1024², and `--mode split` (default) cuts each into four non-overlapping
512 tiles. Validation and testing score every tile; training (`DpgbSceneCropDataset`) groups
the tiles by scene and cuts one random window per scene and epoch out of one of them, so a
sample decodes a quarter of the original. `--fraction 0.25` subsamples for a quick pass.
`--mode resize` downscales instead of tiling, at the cost of one-pixel roads and of
comparability with the tiled results.

---

## What gets logged

Local file store by default (`/kaggle/working/mlruns`, kept as notebook output). Read it with
`MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns`: current MLflow (3.16
tested) refuses a file store without that opt-in, which `tools/mlflow_utils.py` sets for the
pipeline itself. Set a `MLFLOW_TRACKING_URI` **env var or Kaggle Secret** to use a remote
server instead — no code change, and credentials come from the same two places
(`MLFLOW_TRACKING_USERNAME` / `_PASSWORD` / `_TOKEN`).

Per epoch, for train (every epoch) and val (every validation):

- `*_mIoU`, `*_F1`, `*_OA`, and `train_loss` / `val_loss`
- `*_IoU_Road`, `*_F1_Road`, `*_IoU_Background`, `*_F1_Background`
- `epoch_seconds` and the learning rate

With two classes, mIoU is dominated by background, so **`val_IoU_Road` is the number to
watch.** `mIoU` is kept as-is for comparability — see "Metric averaging" below.

At test time (`road_seg_test.py`), on the same run: `test_IoU_*`, `test_F1_*`,
`test_Precision_*`, `test_Recall_*`, plus the topology metrics **`test_clDice`** and
**`test_APLS`**. Those two skeletonise and graph every tile on the CPU, which is why they
are off during training (`VAL_TOPOLOGY_METRICS=1` to enable anyway) and the slow part of
evaluation (`--no-topology` skips them).

Also logged: every scalar config value, `env/*` (GPU, capability, CPU count, torch, CUDA,
whether channels_last is on, this repo's git sha, the upstream model's commit),
`model/params_*`, CPU/GPU/RAM system metrics, the config file as an artifact, a
`training_summary.json`, and a prediction preview grid every 25 epochs under
`val_predictions/` — input, ground truth, prediction, and an error map (green hit, red false
positive, blue missed road).

**Resume behaviour.** The run id is cached in `<weights_path>/mlflow_run.json`. When
`MAX_TIME` stops training early, the run is left open and `last.ckpt` holds the state as of
the latest validation; rerunning picks up both, so a schedule spread over several Kaggle
sessions is one continuous set of curves. Keep `MAX_EPOCH` unchanged when resuming: the
OneCycle schedule is laid out over the original length.

---

## Configuration

[`config/kaggle/ResNet34UNet.py`](config/kaggle/ResNet34UNet.py) reads an env var for every
knob, so the notebook retunes a run without editing files:

```bash
CROP_SIZE=384 GPUS=2 STRATEGY=ddp \
  python train_supervision.py -c config/kaggle/ResNet34UNet.py
```

| variable | default | |
| --- | --- | --- |
| `DEEPGLOBE_ROOT` | `/kaggle/working/data/deepglobe` | prepared split (an attached Dataset works) |
| `OUT_ROOT` | `/kaggle/working` | checkpoints, `mlruns`, results |
| `MAX_EPOCH` | `300` | a multiple of `CHECK_VAL_EVERY_N_EPOCH`; keep it fixed across resumed sessions |
| `CROP_SIZE` | `256` | training window per scene and epoch: a multiple of 64, at most 512 |
| `TRAIN_BATCH_SIZE` / `VAL_BATCH_SIZE` | `16` / `16` | per GPU |
| `LR` / `WEIGHT_DECAY` | `1e-3` / `1e-4` | upstream's; `LR` is the OneCycle peak |
| `LOSS` | `ce_dice` | `ce` for upstream's plain cross-entropy |
| `CHECK_VAL_EVERY_N_EPOCH` | `5` | checkpoints are written after each validation |
| `GPUS` / `STRATEGY` | `1` / `auto` | `2` / `ddp` on a T4 x2 (the notebook sets both) |
| `NUM_WORKERS` | `2` | DataLoader workers per GPU (the notebook sets cores ÷ GPUs) |
| `PRECISION` | `16-mixed` | `32-true` for full precision, in training and evaluation |
| `CHANNELS_LAST` | `auto` | on for compute ≥ 7.0; `0` / `1` to force |
| `DETERMINISTIC` | `0` | `1` for repeatable cuDNN kernels |
| `MAX_TIME` | `00:10:30:00` | `DD:HH:MM:SS` wall-clock stop; `MAX_TIME=` disables it |
| `MONITOR` | `val_mIoU` | `val_IoU_Road` to select checkpoints on road IoU |
| `LIMIT_TRAIN_BATCHES` / `LIMIT_VAL_BATCHES` | `1.0` | fractions or batch counts turn the config into a smoke test |
| `PROGRESS_BAR_REFRESH_RATE` | `25` | steps per redraw; `0` hides the bar |
| `GRADIENT_CLIP_VAL` | `0` (off) | set `1.0` if fp16 sends the loss to NaN |
| `RESNET34_UNET_DIR` | `third_party/resnet34-unet` | another checkout of the upstream repo |

---

## Deviations from upstream

From [GohVh/resnet34-unet](https://github.com/GohVh/resnet34-unet) at `af46a13`.

**Unchanged:** `UnetResnet34` itself, imported from the submodule rather than copied or
edited; its ImageNet-pretrained torchvision ResNet34 encoder; AdamW at lr 1e-3 with weight
decay 1e-4 under OneCycleLR stepped every batch; ImageNet input normalisation.

**Changed, and why:**

- **Dataset**: DeepGlobe roads with two classes (road = 0, background = 1) instead of the
  Semantic Drone Dataset's 24 classes resized to 768×1024.
- **Epoch**: one random `CROP_SIZE` window per scene — see "Training speed".
- **Loss**: cross-entropy (label smoothing 0.05) plus Dice by default, because roads are
  ~4% of the pixels. It gives the same gradients as the RFM-UNet runs' `EdgeLoss`, whose
  edge term is computed from an argmax and so never contributes a gradient. `LOSS=ce`
  restores upstream's plain cross-entropy.
- **Augmentation**: flips, 90° rotations and brightness/contrast — the set the RFM-UNet runs
  used, so a comparison isolates the network — instead of upstream's flips,
  GridDistortion, brightness/contrast and GaussNoise.
- **Validation** is un-augmented, so it is the same measurement every time; upstream's
  validation loader applies random horizontal flips and GridDistortion.
- **No early stopping**: all 300 epochs run and the best checkpoint by `MONITOR` is kept.
  Upstream's `STOP_EPOCH` check compares against a loss that is never updated from its
  initial infinity, so on a fresh run it never fires either.
- **Metrics** come from a dataset-level confusion matrix (plus clDice and APLS at test
  time) rather than upstream's mean of per-batch (training) or per-image (test) mIoU.
- Batch 16 per GPU instead of 3; mixed precision, channels_last and DDP; MLflow instead of
  wandb.

**Metric averaging, unchanged but worth knowing.** `train_supervision.py` keys its
averaging off substrings of `log_name`, and the DeepGlobe config's `dpgb_log` matches none
of the foreground-only datasets, so `mIoU`/`F1` average road **and** background — as in the
RFM-UNet runs, so the numbers stay comparable. `*_IoU_Road` is logged separately so the
road-only figure is always available; `road_seg_test.py` also logs `test_mIoU_foreground`.

---

## Layout

```
resnet34unet-kaggle.ipynb       Kaggle runner: probe -> code + model -> data -> smoke -> train -> eval
train_supervision.py            training entry point (Lightning + MLflow)
road_seg_test.py                evaluation + mask export (TTA, clDice, APLS)
config/kaggle/ResNet34UNet.py   DeepGlobe on a Kaggle GPU, env-var driven
third_party/resnet34-unet/      GohVh/resnet34-unet, git submodule pinned at af46a13
geoseg/models/resnet34_unet.py  loads UnetResnet34 from the submodule
geoseg/losses/                  cross-entropy, Dice and the rest of the GeoSeg loss zoo
geoseg/datasets/                dpgb (DeepGlobe: scene crops to train, tiles to score), mass, CHN6
tools/prepare_deepglobe.py      Kaggle dataset -> 512 tiles, masks re-encoded, verified
tools/mlflow_utils.py           tracking URI resolution, secrets, params, run-id resume
tools/callbacks.py              epoch timer and ETA, validation prediction previews
tools/{cfg,metric,utils}.py     GeoSeg helpers (metric.py: topology switch + DDP merge)
```

## Credit

ResNet34-UNet by [GohVh](https://github.com/GohVh/resnet34-unet). That repository has no
licence file, so it is referenced here as a submodule rather than redistributed. The
architecture combines ResNet ([He et al., 2015](https://arxiv.org/abs/1512.03385)) and
U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)). Training scaffolding
from [GeoSeg](https://github.com/WangLibo1995/GeoSeg) by way of
[RFM-UNet](https://github.com/FF7CA/RFMUNet) (MIT).
